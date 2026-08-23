//! Measure the ceiling of selecting among the existing split-radiance filters.
//!
//! The oracle chooses one of the unfiltered lobe or five à-trous results over
//! blocks, against independently rendered lobe references. It deliberately
//! does not add a model or runtime contract: if this target has no useful
//! ceiling, there is no reason to make the shipping path more complicated.

use std::path::{Path, PathBuf};

use ommatidia::batch::{self, Crop, SPLIT_FILTER_SCALES};
use ommatidia::dataset::{Layout, Plane, Reader, Sample};
use ommatidia::model::GuideConfig;

struct Args {
    data: PathBuf,
    reference_data: Option<PathBuf>,
    out: Option<PathBuf>,
    blocks: Vec<usize>,
    max_sequences: Option<usize>,
}

fn usage() -> &'static str {
    "measure block-constrained per-lobe filter-scale headroom\n\n\
usage: lobe-scale-oracle --data PATH [options]\n\n\
  --reference-data PATH  matched dataset carrying cleaner HR lobe references\n\
  --blocks N,N,...       output-pixel block sizes [4,8,16,32]\n\
  --max-sequences N      score only the first N matched sequences\n\
  --out DIR              write a mature-frame comparison\n\
  -h, --help             show this text"
}

fn parse_args() -> Result<Args, String> {
    let mut data = None;
    let mut reference_data = None;
    let mut out = None;
    let mut blocks = vec![4, 8, 16, 32];
    let mut max_sequences = None;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        let mut value = || args.next().ok_or_else(|| format!("{arg} needs a value"));
        match arg.as_str() {
            "--data" => data = Some(PathBuf::from(value()?)),
            "--reference-data" => reference_data = Some(PathBuf::from(value()?)),
            "--out" => out = Some(PathBuf::from(value()?)),
            "--blocks" => {
                let text = value()?;
                blocks = text
                    .split(',')
                    .map(|part| {
                        part.parse::<usize>()
                            .map_err(|e| format!("invalid block size {part:?}: {e}"))
                    })
                    .collect::<Result<_, _>>()?;
            }
            "--max-sequences" => {
                max_sequences = Some(
                    value()?
                        .parse::<usize>()
                        .map_err(|e| format!("--max-sequences: {e}"))?,
                );
            }
            "-h" | "--help" => {
                println!("{}", usage());
                std::process::exit(0);
            }
            other => return Err(format!("unknown option {other:?}\n\n{}", usage())),
        }
    }
    if blocks.is_empty() || blocks.contains(&0) {
        return Err("block sizes must be positive".into());
    }
    if max_sequences == Some(0) {
        return Err("--max-sequences must be positive".into());
    }
    Ok(Args {
        data: data.ok_or_else(|| format!("--data is required\n\n{}", usage()))?,
        reference_data,
        out,
        blocks,
        max_sequences,
    })
}

fn interleaved_hr(sample: &Sample, layout: &Layout, plane: Plane) -> Vec<f32> {
    let texels = layout.hr_texels();
    let mut out = vec![0.0; texels * 3];
    for component in 0..3 {
        let source = sample
            .hr_channel(layout, plane, component)
            .unwrap_or_else(|| panic!("reference has no HR {plane:?}"));
        for (index, value) in source.iter().enumerate() {
            out[index * 3 + component] = value.to_f32();
        }
    }
    out
}

fn compose(
    diffuse: &[f32],
    specular: &[f32],
    emissive: &[f32],
    sample: &Sample,
    layout: &Layout,
) -> Vec<f32> {
    let texels = layout.hr_texels();
    assert_eq!(diffuse.len(), texels * 3);
    let mut out = vec![0.0; texels * 3];
    for component in 0..3 {
        let albedo = sample
            .hr_channel(layout, Plane::DiffuseAlbedo, component)
            .expect("input has no HR diffuse albedo");
        for (index, albedo) in albedo.iter().enumerate() {
            let offset = index * 3 + component;
            out[offset] = albedo.to_f32() * diffuse[offset] + specular[offset] + emissive[offset];
        }
    }
    out
}

struct BlockSelection {
    image: Vec<f32>,
    histogram: Vec<usize>,
    labels: Vec<usize>,
}

fn choose_blocks(
    candidates: &[Vec<f32>],
    reference: &[f32],
    width: usize,
    height: usize,
    block: usize,
) -> BlockSelection {
    let mut out = vec![0.0; reference.len()];
    let mut histogram = vec![0; candidates.len()];
    let mut labels = Vec::with_capacity(width.div_ceil(block) * height.div_ceil(block));
    for by in (0..height).step_by(block) {
        for bx in (0..width).step_by(block) {
            let x_end = (bx + block).min(width);
            let y_end = (by + block).min(height);
            let mut errors = vec![0.0f64; candidates.len()];
            for y in by..y_end {
                for x in bx..x_end {
                    let offset = (y * width + x) * 3;
                    for component in 0..3 {
                        let target = ommatidia::transform::compress(reference[offset + component]);
                        for scale in 0..candidates.len() {
                            let value = ommatidia::transform::compress(
                                candidates[scale][offset + component],
                            );
                            errors[scale] += f64::from((value - target).powi(2));
                        }
                    }
                }
            }
            let scale = errors
                .iter()
                .enumerate()
                .min_by(|a, b| a.1.total_cmp(b.1))
                .unwrap()
                .0;
            labels.push(scale);
            histogram[scale] += (x_end - bx) * (y_end - by);
            for y in by..y_end {
                for x in bx..x_end {
                    let offset = (y * width + x) * 3;
                    out[offset..offset + 3].copy_from_slice(&candidates[scale][offset..offset + 3]);
                }
            }
        }
    }
    BlockSelection {
        image: out,
        histogram,
        labels,
    }
}

fn apply_blocks(
    candidates: &[Vec<f32>],
    width: usize,
    height: usize,
    block: usize,
    labels: &[usize],
) -> Vec<f32> {
    assert_eq!(labels.len(), width.div_ceil(block) * height.div_ceil(block));
    let mut out = vec![0.0; width * height * 3];
    let mut label = 0;
    for by in (0..height).step_by(block) {
        for bx in (0..width).step_by(block) {
            let candidate = &candidates[labels[label]];
            label += 1;
            for y in by..(by + block).min(height) {
                for x in bx..(bx + block).min(width) {
                    let offset = (y * width + x) * 3;
                    out[offset..offset + 3].copy_from_slice(&candidate[offset..offset + 3]);
                }
            }
        }
    }
    out
}

#[derive(Default)]
struct Scores {
    error: f64,
    ssim: f64,
    relative: f64,
    detail: f64,
    reference_detail: f64,
    low_frequency: f64,
    energy: f64,
    reference_energy: f64,
    frames: usize,
}

fn mean_luminance(image: &[f32]) -> f64 {
    image
        .chunks_exact(3)
        .map(|rgb| 0.2126 * rgb[0] as f64 + 0.7152 * rgb[1] as f64 + 0.0722 * rgb[2] as f64)
        .sum::<f64>()
        / (image.len() / 3).max(1) as f64
}

impl Scores {
    fn add(&mut self, image: &[f32], reference: &[f32], width: usize, height: usize) {
        self.error += f64::from(ommatidia::metrics::error(image, reference));
        self.ssim += f64::from(ommatidia::metrics::ssim(image, reference, width, height));
        self.relative += ommatidia::metrics::relative_error(image, reference);
        self.detail += ommatidia::metrics::detail(image, width, height);
        self.reference_detail += ommatidia::metrics::detail(reference, width, height);
        self.low_frequency +=
            ommatidia::metrics::low_frequency_error(image, reference, width, height, 16);
        self.energy += mean_luminance(image);
        self.reference_energy += mean_luminance(reference);
        self.frames += 1;
    }

    fn line(&self, name: &str) -> String {
        let count = self.frames.max(1) as f64;
        let mse = self.error / count;
        let low = self.low_frequency / count;
        format!(
            "{name:<12} PSNR {:>5.2} dB, SSIM {:.4}, relMSE {:.5}, detail {:>5.1}%, \
             energy {:.3}, LF PSNR {:>5.2} dB",
            -10.0 * mse.log10(),
            self.ssim / count,
            self.relative / count,
            100.0 * self.detail / self.reference_detail,
            self.energy / self.reference_energy,
            -10.0 * low.log10(),
        )
    }
}

fn write_png(path: &Path, image: &[f32], width: usize, height: usize) -> Result<(), String> {
    let mut bytes = Vec::with_capacity(width * height * 4);
    for texel in image.chunks_exact(3) {
        for &linear in texel {
            let mapped = ommatidia::transform::compress(linear);
            let encoded = if mapped <= 0.003_130_8 {
                12.92 * mapped
            } else {
                1.055 * mapped.powf(1.0 / 2.4) - 0.055
            };
            bytes.push((encoded.clamp(0.0, 1.0) * 255.0 + 0.5) as u8);
        }
        bytes.push(255);
    }
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    let file = std::fs::File::create(path).map_err(|e| e.to_string())?;
    let mut encoder = png::Encoder::new(file, width as u32, height as u32);
    encoder.set_color(png::ColorType::Rgba);
    encoder.set_depth(png::BitDepth::Eight);
    encoder
        .write_header()
        .and_then(|mut writer| writer.write_image_data(&bytes))
        .map_err(|e| e.to_string())
}

fn validate_layout(data: &Layout, reference: &Layout) -> Result<(), String> {
    if (data.scale, data.lr_width, data.lr_height)
        != (reference.scale, reference.lr_width, reference.lr_height)
    {
        return Err("input and reference extents differ".into());
    }
    for plane in [
        Plane::Color,
        Plane::DiffuseIllumination,
        Plane::SpecularRadiance,
    ] {
        if !reference.hr_planes.contains(plane) {
            return Err(format!("reference has no HR {plane:?}"));
        }
    }
    for plane in [
        Plane::DiffuseIllumination,
        Plane::SpecularRadiance,
        Plane::EmissiveRadiance,
        Plane::Depth,
        Plane::Normal,
        Plane::DiffuseAlbedo,
    ] {
        if !data.lr_planes.contains(plane) {
            return Err(format!("input has no LR {plane:?}"));
        }
    }
    for plane in [
        Plane::Color,
        Plane::Depth,
        Plane::Normal,
        Plane::DiffuseAlbedo,
        Plane::Roughness,
    ] {
        if !data.hr_planes.contains(plane) {
            return Err(format!("input has no HR {plane:?}"));
        }
    }
    Ok(())
}

fn accumulate_histogram(
    total: &mut [usize; SPLIT_FILTER_SCALES],
    frame: [usize; SPLIT_FILTER_SCALES],
) {
    for (total, frame) in total.iter_mut().zip(frame) {
        *total += frame;
    }
}

fn histogram_line(histogram: &[usize; SPLIT_FILTER_SCALES]) -> String {
    let total = histogram.iter().sum::<usize>().max(1) as f64;
    histogram
        .iter()
        .enumerate()
        .map(|(scale, &count)| format!("{scale}: {:.1}%", 100.0 * count as f64 / total))
        .collect::<Vec<_>>()
        .join(", ")
}

fn run(args: Args) -> Result<(), String> {
    let mut reader = Reader::open(&args.data).map_err(|e| e.to_string())?;
    let layout = *reader.layout();
    let data_sequences = reader.len() / reader.sequence_length();
    let mut reference_reader = args
        .reference_data
        .as_ref()
        .map(|path| Reader::open(path).map_err(|e| e.to_string()))
        .transpose()?;
    let reference_layout = reference_reader
        .as_ref()
        .map_or(layout, |reference| *reference.layout());
    validate_layout(&layout, &reference_layout)?;
    let reference_sequence = reference_reader
        .as_ref()
        .map_or(reader.sequence_length(), Reader::sequence_length);
    if reference_sequence != 1 && reference_sequence != reader.sequence_length() {
        return Err("reference sequence length must be one or match the input".into());
    }
    let reference_sequences = reference_reader
        .as_ref()
        .map_or(data_sequences, |reference| {
            reference.len() / reference.sequence_length()
        });
    let sequence_count = args
        .max_sequences
        .unwrap_or(usize::MAX)
        .min(data_sequences)
        .min(reference_sequences);
    if sequence_count == 0 || reader.sequence_length() < 2 {
        return Err("the matched data contains no temporal sequences".into());
    }
    println!(
        "{} matched sequences, {} frames each, {}x{} -> {}x{}",
        sequence_count,
        reader.sequence_length(),
        layout.lr_width,
        layout.lr_height,
        layout.hr_width(),
        layout.hr_height(),
    );

    let crop = Crop {
        x: 0,
        y: 0,
        tile: layout.lr_width,
    };
    let width = layout.hr_width() as usize;
    let height = layout.hr_height() as usize;
    let mut fixed_scores = Scores::default();
    let mut lobe_scores: Vec<_> = args.blocks.iter().map(|_| Scores::default()).collect();
    let mut lobe_lagged_scores: Vec<_> = args.blocks.iter().map(|_| Scores::default()).collect();
    let mut diffuse_histograms = vec![[0; SPLIT_FILTER_SCALES]; args.blocks.len()];
    let mut specular_histograms = vec![[0; SPLIT_FILTER_SCALES]; args.blocks.len()];
    let temporal = ommatidia::temporal::Config {
        frames: reader.sequence_length() as u32,
        rejection: ommatidia::temporal::RejectionConfig::default(),
        features: ommatidia::temporal::Features::PhaseLobes,
        unrejected_tap: false,
        previous_output: false,
    };
    let mut frames = 0usize;
    let started = std::time::Instant::now();
    for sequence in 0..sequence_count {
        let mut previous_lobe_labels: Vec<Option<[Vec<usize>; 2]>> =
            args.blocks.iter().map(|_| None).collect();
        for frame in 1..reader.sequence_length() {
            let index = sequence * reader.sequence_length() + frame;
            let prepared = ommatidia::temporal::prepare(&mut reader, index, temporal)
                .map_err(|e| e.to_string())?;
            let reference_index =
                sequence * reference_sequence + if reference_sequence == 1 { 0 } else { frame };
            let reference = if let Some(reference_reader) = &mut reference_reader {
                reference_reader
                    .sample(reference_index)
                    .map_err(|e| e.to_string())?
            } else {
                reader.sample(index).map_err(|e| e.to_string())?
            };
            let reference_color = interleaved_hr(&reference, &reference_layout, Plane::Color);
            let reference_diffuse =
                interleaved_hr(&reference, &reference_layout, Plane::DiffuseIllumination);
            let reference_specular =
                interleaved_hr(&reference, &reference_layout, Plane::SpecularRadiance);
            let sample = &prepared.sample;
            let fixed =
                batch::high_resolution_split_base(sample, &layout, crop, GuideConfig::TUNED);
            fixed_scores.add(&fixed, &reference_color, width, height);
            let candidates = batch::high_resolution_split_filter_candidates(
                sample,
                &layout,
                crop,
                GuideConfig::TUNED,
            );
            for (oracle_index, &block) in args.blocks.iter().enumerate() {
                let diffuse_selection = choose_blocks(
                    &candidates.diffuse,
                    &reference_diffuse,
                    width,
                    height,
                    block,
                );
                let specular_selection = choose_blocks(
                    &candidates.specular,
                    &reference_specular,
                    width,
                    height,
                    block,
                );
                if let Some([diffuse_labels, specular_labels]) = &previous_lobe_labels[oracle_index]
                {
                    let diffuse =
                        apply_blocks(&candidates.diffuse, width, height, block, diffuse_labels);
                    let specular =
                        apply_blocks(&candidates.specular, width, height, block, specular_labels);
                    let lagged =
                        compose(&diffuse, &specular, &candidates.emissive, sample, &layout);
                    lobe_lagged_scores[oracle_index].add(&lagged, &reference_color, width, height);
                }
                previous_lobe_labels[oracle_index] = Some([
                    diffuse_selection.labels.clone(),
                    specular_selection.labels.clone(),
                ]);
                let lobe_oracle = compose(
                    &diffuse_selection.image,
                    &specular_selection.image,
                    &candidates.emissive,
                    sample,
                    &layout,
                );
                lobe_scores[oracle_index].add(&lobe_oracle, &reference_color, width, height);
                let diffuse_histogram: [usize; SPLIT_FILTER_SCALES] =
                    diffuse_selection.histogram.try_into().unwrap();
                let specular_histogram: [usize; SPLIT_FILTER_SCALES] =
                    specular_selection.histogram.try_into().unwrap();
                accumulate_histogram(&mut diffuse_histograms[oracle_index], diffuse_histogram);
                accumulate_histogram(&mut specular_histograms[oracle_index], specular_histogram);
                if sequence == 0
                    && frame + 1 == reader.sequence_length()
                    && let Some(out) = &args.out
                {
                    write_png(
                        &out.join(format!("lobe-oracle-b{block}.png")),
                        &lobe_oracle,
                        width,
                        height,
                    )?;
                }
            }
            if sequence == 0
                && frame + 1 == reader.sequence_length()
                && let Some(out) = &args.out
            {
                write_png(&out.join("fixed.png"), &fixed, width, height)?;
                write_png(&out.join("reference.png"), &reference_color, width, height)?;
            }
            frames += 1;
        }
        println!(
            "  {}/{} sequences ({:.1}s)",
            sequence + 1,
            sequence_count,
            started.elapsed().as_secs_f32(),
        );
    }
    println!("scored {frames} non-reset full frames");
    println!("{}", fixed_scores.line("fixed"));
    for (index, &block) in args.blocks.iter().enumerate() {
        println!("{}", lobe_scores[index].line(&format!("lobe b{block}")));
        println!(
            "{}",
            lobe_lagged_scores[index].line(&format!("lobe lag b{block}"))
        );
        println!(
            "  diffuse scales: {}; specular scales: {}",
            histogram_line(&diffuse_histograms[index]),
            histogram_line(&specular_histograms[index]),
        );
    }
    Ok(())
}

fn main() {
    let result = parse_args().and_then(run);
    if let Err(message) = result {
        eprintln!("{message}");
        std::process::exit(2);
    }
}
