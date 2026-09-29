//! Immutable, read-only mapped captures. Only selected crop rows are expanded.
//!
//! Capture files must not be modified or truncated while mapped. The trainer
//! never writes captures; content hashes bind checkpoints to the ordered inputs.
use crate::{Result, open_capture};
use half::f16;
use ommatidia::{
    dataset::{HEADER_SIZE, Layout, Plane, Sample},
    transport::{Config, Frame, Target},
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::path::Path;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Identity {
    pub sha256: String,
    pub provenance_sha256: String,
    pub bytes: usize,
}

pub struct Capture {
    map: memmap2::Mmap,
    pub layout: Layout,
    pub frames: usize,
    pub length: usize,
    pub provenance: serde_json::Value,
    pub identity: Identity,
}

impl Capture {
    pub fn open(path: &Path, config: Config) -> Result<Self> {
        let (reader, provenance) = open_capture(path)?;
        let layout = *reader.layout();
        config.validate([layout.lr_width, layout.lr_height])?;
        if layout.scale != config.scale {
            return Err("capture/model scale mismatch".into());
        }
        let file = std::fs::File::open(path)?;
        // SAFETY: captures are immutable inputs for the lifetime of this map.
        // This contract also applies to other processes running on the host.
        let map = unsafe { memmap2::Mmap::map(&file)? };
        let identity = Identity {
            sha256: format!("{:x}", Sha256::digest(&map)),
            provenance_sha256: format!(
                "{:x}",
                Sha256::digest(std::fs::read(path.with_extension("transport.json"))?)
            ),
            bytes: map.len(),
        };
        let capture = Self {
            map,
            layout,
            frames: reader.len(),
            length: reader.sequence_length(),
            provenance,
            identity,
        };
        // Validate all required planes and their units using the ordinary decoder.
        capture.crop(0, [0, 0], [4.max(1 << (config.levels - 1)); 2], 1.0, config)?;
        Ok(capture)
    }

    fn record(&self, index: usize) -> Result<&[f16]> {
        if index >= self.frames {
            return Err("frame outside capture".into());
        }
        let len = self.layout.record_len() * 2;
        let start = HEADER_SIZE + index * len;
        // .omd is little-endian, f16; decode explicitly on big-endian machines.
        if cfg!(target_endian = "big") {
            return Err("mapped captures require little-endian host".into());
        }
        Ok(bytemuck::try_cast_slice(&self.map[start..start + len])
            .map_err(|e| format!("unaligned mapped record: {e:?}"))?)
    }

    pub fn crop(
        &self,
        index: usize,
        origin: [u32; 2],
        low: [u32; 2],
        gain: f32,
        config: Config,
    ) -> Result<(Frame, Target)> {
        config.validate(low)?;
        if !gain.is_finite()
            || gain <= 0.0
            || origin[0] + low[0] > self.layout.lr_width
            || origin[1] + low[1] > self.layout.lr_height
        {
            return Err("invalid crop or radiance gain".into());
        }
        let record = self.record(index)?;
        let gather = |source: &[f16],
                      extent: [u32; 2],
                      origin: [u32; 2],
                      size: [u32; 2],
                      channels: usize| {
            let mut values = Vec::with_capacity((size[0] * size[1]) as usize * channels);
            for c in 0..channels {
                for y in origin[1]..origin[1] + size[1] {
                    let offset = ((c as u32 * extent[1] + y) * extent[0] + origin[0]) as usize;
                    values.extend_from_slice(&source[offset..offset + size[0] as usize]);
                }
            }
            values
        };
        let s = config.scale;
        let sample = Sample {
            lr: gather(
                &record[..self.layout.lr_len()],
                [self.layout.lr_width, self.layout.lr_height],
                origin,
                low,
                self.layout.lr_planes.channels(),
            ),
            hr: gather(
                &record[self.layout.lr_len()..],
                [self.layout.hr_width(), self.layout.hr_height()],
                origin.map(|x| x * s),
                low.map(|x| x * s),
                self.layout.hr_planes.channels(),
            ),
        };
        let layout = Layout {
            lr_width: low[0],
            lr_height: low[1],
            ..self.layout
        };
        let mut frame = Frame::from_sample(&sample, layout, config)?;
        let mut target = Target::from_sample(&sample, layout, config)?;
        if target
            .lobes
            .iter()
            .chain(&target.rgb)
            .any(|v| !v.is_finite() || *v < 0.0)
        {
            return Err(format!("invalid reference in frame {index}").into());
        }
        // Gain is applied after f16 expansion, never requantized. Exposure is 1.
        for ray in &mut frame.rays {
            for v in ray.diffuse[..3].iter_mut().chain(&mut ray.specular[..3]) {
                *v *= gain;
            }
        }
        for surface in &mut frame.surfaces {
            for v in &mut surface.emission[..3] {
                *v *= gain;
            }
        }
        for v in target.lobes.iter_mut().chain(&mut target.rgb) {
            *v *= gain;
        }
        Ok((frame, target))
    }

    fn reference_sum(&self) -> Result<([f64; 6], usize)> {
        let n = self.layout.hr_texels();
        let offsets = [
            Plane::DiffuseIllumination,
            Plane::SpecularRadiance,
            Plane::Color,
        ]
        .map(|p| {
            self.layout
                .hr_planes
                .channel_offset(p)
                .ok_or("missing reference plane")
        });
        let offsets = offsets
            .into_iter()
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let mut sums = [0.0; 6];
        for index in 0..self.frames {
            let high = &self.record(index)?[self.layout.lr_len()..];
            let mut color_sum = 0.0;
            for (plane, offset) in offsets.iter().enumerate() {
                for c in 0..3 {
                    let mut sum = 0.0;
                    for v in &high[(offset + c) * n..(offset + c + 1) * n] {
                        let v = v.to_f32();
                        if !v.is_finite() || v < 0.0 {
                            return Err(format!("invalid reference at frame {index}").into());
                        }
                        sum += f64::from(v);
                    }
                    if plane < 2 {
                        sums[plane * 3 + c] += sum;
                    } else {
                        color_sum += sum;
                    }
                }
            }
            if color_sum <= 1e-6 {
                return Err(format!("entirely black reference at frame {index}").into());
            }
        }
        Ok((sums, n * self.frames))
    }
}

pub struct Corpus {
    pub captures: Vec<Capture>,
    pub low: [u32; 2],
    pub length: usize,
    pub sequences: Vec<(usize, usize)>,
    pub provenance: serde_json::Value,
}
impl Corpus {
    pub fn open(paths: &[std::path::PathBuf], config: Config) -> Result<Self> {
        let captures = paths
            .iter()
            .map(|p| Capture::open(p, config))
            .collect::<Result<Vec<_>>>()?;
        let first = captures.first().ok_or("empty capture list")?;
        let low = [first.layout.lr_width, first.layout.lr_height];
        let length = first.length;
        if captures
            .iter()
            .any(|c| c.length != length || [c.layout.lr_width, c.layout.lr_height] != low)
        {
            return Err("capture sequence/extent mismatch".into());
        }
        let sequences = captures
            .iter()
            .enumerate()
            .flat_map(|(c, capture)| (0..capture.frames / length).map(move |i| (c, i * length)))
            .collect();
        let provenance = serde_json::json!({"captures":captures.iter().map(|c| &c.provenance).collect::<Vec<_>>()});
        Ok(Self {
            captures,
            low,
            length,
            sequences,
            provenance,
        })
    }
    pub fn len(&self) -> usize {
        self.sequences.len() * self.length
    }
    pub fn is_empty(&self) -> bool {
        self.sequences.is_empty()
    }
    pub fn identities(&self) -> Vec<Identity> {
        self.captures.iter().map(|c| c.identity.clone()).collect()
    }
    pub fn crop(
        &self,
        sequence: usize,
        frame: usize,
        origin: [u32; 2],
        low: [u32; 2],
        gain: f32,
        config: Config,
    ) -> Result<(Frame, Target)> {
        let &(capture, first) = self
            .sequences
            .get(sequence)
            .ok_or("sequence outside corpus")?;
        if frame >= self.length {
            return Err("frame outside sequence".into());
        }
        self.captures[capture].crop(first + frame, origin, low, gain, config)
    }
    pub fn decode(&self, index: usize, config: Config) -> Result<(Frame, Target)> {
        self.crop(
            index / self.length,
            index % self.length,
            [0, 0],
            self.low,
            1.0,
            config,
        )
    }
    pub fn means(&self) -> Result<[f64; 6]> {
        let mut sums = [0.0; 6];
        let mut n = 0;
        for capture in &self.captures {
            let (s, pixels) = capture.reference_sum()?;
            for (a, b) in sums.iter_mut().zip(s) {
                *a += b;
            }
            n += pixels;
        }
        Ok(sums.map(|v| v / n as f64))
    }
    pub fn disjoint(&self, other: &Self) -> Result<()> {
        for a in &self.captures {
            for b in &other.captures {
                for key in ["scene_seeds", "family_ids"] {
                    let empty = Vec::new();
                    let ids = |p: &serde_json::Value| p[key].as_array().cloned();
                    let aa = ids(&a.provenance);
                    let bb = ids(&b.provenance);
                    if key == "scene_seeds" && (aa.is_none() || bb.is_none()) {
                        return Err("missing scene-seed provenance".into());
                    }
                    if aa
                        .as_ref()
                        .unwrap_or(&empty)
                        .iter()
                        .any(|v| bb.as_ref().unwrap_or(&empty).contains(v))
                    {
                        return Err(format!("training/evaluation {key} overlap").into());
                    }
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ommatidia::dataset::{ALL_PLANES, InputSource, PlaneSet, Reader, Writer};
    #[test]
    fn mapped_crops_match_original_decode_gain_and_prefetch_resume() {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let directory =
            std::env::temp_dir().join(format!("ommatidia-mapped-{}-{nonce}", std::process::id()));
        std::fs::create_dir(&directory).unwrap();
        let path = directory.join("capture.omd");
        let planes = ALL_PLANES
            .into_iter()
            .fold(PlaneSet::new(), |set, p| set.with(p));
        let layout = Layout {
            lr_width: 8,
            lr_height: 8,
            scale: 2,
            lr_source: InputSource::PathTrace,
            lr_planes: planes,
            hr_planes: planes,
        };
        let mut writer = Writer::create_sequence(&path, layout, 16).unwrap();
        for frame in 0..32 {
            let values = |len| {
                (0..len)
                    .map(|i| {
                        f16::from_f32(0.125 + (i % 31) as f32 * 0.03125 + frame as f32 * 0.015625)
                    })
                    .collect()
            };
            writer
                .write(&Sample {
                    lr: values(layout.lr_len()),
                    hr: values(layout.hr_len()),
                })
                .unwrap();
        }
        writer.finish().unwrap();
        let sidecar = path.with_extension("transport.json");
        let provenance = serde_json::json!({"matching_path_depth":true,"input_estimator":"independent-paths","records":32,"scene_seeds":[7,8],"family_ids":["chair"]});
        std::fs::write(&sidecar, serde_json::to_vec(&provenance).unwrap()).unwrap();
        let config = Config {
            levels: 3,
            ..Config::default()
        }; // 4x4 crop fixture.
        let corpus =
            std::sync::Arc::new(Corpus::open(std::slice::from_ref(&path), config).unwrap());
        let sample = Reader::open(&path).unwrap().sample(19).unwrap();
        let original = Frame::from_sample(&sample, layout, config).unwrap();
        let original_target = Target::from_sample(&sample, layout, config).unwrap();
        let (full, target) = corpus.decode(19, config).unwrap();
        assert_eq!(
            bytemuck::cast_slice::<_, u8>(&original.rays),
            bytemuck::cast_slice::<_, u8>(&full.rays)
        );
        assert_eq!(original.surfaces, full.surfaces);
        assert_eq!(original_target.lobes, target.lobes);
        assert_eq!(original_target.rgb, target.rgb);
        let (cropped, target) = corpus.crop(1, 3, [3, 2], [4, 4], 2.0, config).unwrap();
        assert_eq!(cropped.exposure, 1.0);
        for y in 0..4 {
            for x in 0..4 {
                let src = original.rays[(y + 2) * 8 + x + 3];
                let dst = cropped.rays[y * 4 + x];
                assert_eq!(src.normal_depth, dst.normal_depth);
                for c in 0..3 {
                    assert_eq!(src.diffuse[c] * 2.0, dst.diffuse[c]);
                    assert_eq!(src.specular[c] * 2.0, dst.specular[c]);
                }
            }
        }
        for y in 0..8 {
            for x in 0..8 {
                let src = original.surfaces[(y + 4) * 16 + x + 6];
                let dst = cropped.surfaces[y * 8 + x];
                assert_eq!(src.motion, dst.motion);
                assert_eq!(src.albedo_roughness, dst.albedo_roughness);
                for c in 0..3 {
                    assert_eq!(src.emission[c] * 2.0, dst.emission[c]);
                    assert_eq!(
                        target.rgb[(y * 8 + x) * 3 + c],
                        original_target.rgb[((y + 4) * 16 + x + 6) * 3 + c] * 2.0
                    );
                }
                for c in 0..6 {
                    assert_eq!(
                        target.lobes[config.index([4; 2], c, x, y)],
                        original_target.lobes[config.index([8; 2], c, x + 6, y + 4)] * 2.0
                    );
                }
            }
        }
        assert!(
            corpus
                .means()
                .unwrap()
                .iter()
                .all(|v| v.is_finite() && *v > 0.0)
        );
        assert!(corpus.disjoint(&corpus).is_err());
        let sampler = crate::sampler::Sampler::new(31, 2, [4; 2], 4, [8; 2], 2, 16).unwrap();
        let worker = crate::sampler::Prefetch::new(std::sync::Arc::clone(&corpus), config, sampler);
        let first = worker.receive().unwrap();
        let restored = crate::sampler::Prefetch::new(
            std::sync::Arc::clone(&corpus),
            config,
            serde_json::from_slice(&serde_json::to_vec(&first.next_sampler).unwrap()).unwrap(),
        );
        for _ in 0..5 {
            let a = worker.receive().unwrap();
            let b = restored.receive().unwrap();
            assert_eq!(a.windows, b.windows);
            for (a, b) in a.frames.iter().flatten().zip(b.frames.iter().flatten()) {
                assert_eq!(a.1.lobes, b.1.lobes);
                assert_eq!(a.1.rgb, b.1.rgb);
            }
        }
        drop(worker);
        drop(restored);
        drop(corpus);
        std::fs::remove_file(path).unwrap();
        std::fs::remove_file(sidecar).unwrap();
        std::fs::remove_dir(directory).unwrap();
    }
}
