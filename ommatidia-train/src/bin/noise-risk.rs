//! Same clean scenes, disjoint path streams. A construction diagnostic, not a new-scene audit.
use ommatidia::{
    dataset,
    transport::{CANDIDATES, Config, Frame, SCALES, Target, native, noise, oracle},
};
use std::{
    path::{Path, PathBuf},
    sync::Arc,
};
type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
const METHODS: [&str; 6] = [
    "learned",
    "fixed-prior",
    "observable-risk",
    "cross-noise-risk",
    "single-oracle",
    "convex-oracle",
];
struct Capture {
    frames: Vec<(Frame, Target)>,
    length: usize,
    offset: u64,
    frames_per_input: u64,
    provenance: serde_json::Value,
}
fn load(path: &Path, c: Config) -> Result<Capture> {
    let provenance: serde_json::Value =
        serde_json::from_slice(&std::fs::read(path.with_extension("transport.json"))?)?;
    if provenance["matching_path_depth"] != true
        || provenance["input_estimator"] != "independent-paths"
        || provenance["reference_from"] != serde_json::Value::Null
    {
        return Err("need fresh matched independent-path captures, not copied references".into());
    }
    let offset = provenance["input_sample_offset"]
        .as_u64()
        .ok_or("missing input path offset")?;
    let frames_per_input = provenance["input_frames"]
        .as_u64()
        .filter(|v| *v > 0)
        .ok_or("missing input path count")?;
    let mut reader = dataset::Reader::open(path)?;
    let layout = *reader.layout();
    let length = reader.sequence_length();
    if reader.is_empty() || reader.len() as u64 != provenance["records"].as_u64().unwrap_or(0) {
        return Err("empty or invalid capture".into());
    }
    let mut frames = Vec::new();
    for i in 0..reader.len() {
        let s = reader.sample(i)?;
        frames.push((
            Frame::from_sample(&s, layout, c)?,
            Target::from_sample(&s, layout, c)?,
        ));
    }
    Ok(Capture {
        frames,
        length,
        offset,
        frames_per_input,
        provenance,
    })
}
fn validate(captures: &[Capture]) -> Result<()> {
    let base = &captures[0];
    for (a, c) in captures.iter().enumerate() {
        if c.frames.len() != base.frames.len()
            || c.length != base.length
            || c.frames_per_input != base.frames_per_input
        {
            return Err("capture layout/sequence mismatch".into());
        }
        let end = c
            .offset
            .checked_add(
                (c.frames.len() as u64)
                    .checked_mul(c.frames_per_input)
                    .ok_or("path range overflow")?,
            )
            .ok_or("path range overflow")?;
        let ref_start = c.provenance["reference_sample_offset"]
            .as_u64()
            .ok_or("missing reference offset")?;
        if end > ref_start {
            return Err("reference stream must start after every retained noisy stream".into());
        }
        for old in &captures[..a] {
            let old_end = old.offset + old.frames.len() as u64 * old.frames_per_input;
            if c.offset < old_end && old.offset < end {
                return Err("overlapping path streams".into());
            }
        }
        for key in [
            "scene_seeds",
            "family_ids",
            "capture_seed",
            "input_max_bounces",
            "reference_max_bounces",
            "light_motion",
            "canonical_frames",
            "reference_sample_offset",
        ] {
            if c.provenance[key] != base.provenance[key] {
                return Err(format!("unequal provenance: {key}").into());
            }
        }
        let mut changed = false;
        for ((f, t), (b, bt)) in c.frames.iter().zip(&base.frames) {
            if f.low != b.low
                || f.jitter != b.jitter
                || f.rays.len() != b.rays.len()
                || f.surfaces.len() != b.surfaces.len()
                || t.lobes != bt.lobes
                || t.rgb != bt.rgb
            {
                return Err(
                    "clean targets or dimensions differ; not a paired noise capture".into(),
                );
            }
            for (x, y) in f.rays.iter().zip(&b.rays) {
                if x.normal_depth != y.normal_depth || x.albedo_roughness != y.albedo_roughness {
                    return Err("noisy capture changed LR geometry".into());
                }
                changed |= x.diffuse != y.diffuse || x.specular != y.specular;
            }
            for (x, y) in f.surfaces.iter().zip(&b.surfaces) {
                if x.normal_depth != y.normal_depth
                    || x.albedo_roughness != y.albedo_roughness
                    || x.emission != y.emission
                    || x.motion != y.motion
                    || x.specular_motion != y.specular_motion
                {
                    return Err("noisy capture changed HR guides".into());
                }
            }
        }
        if a > 0 && !changed {
            return Err("path offset did not change sparse radiance".into());
        }
    }
    Ok(())
}
fn points(s: &oracle::Candidates, l: usize, i: usize, n: usize) -> [[f32; 3]; CANDIDATES] {
    std::array::from_fn(|k| {
        std::array::from_fn(|c| {
            if k == SCALES {
                s.history[(3 * l + c) * n + i]
            } else {
                s.spatial[k * 6 * n + (3 * l + c) * n + i]
            }
        })
    })
}
fn selected(
    s: &oracle::Candidates,
    p: &[[f32; 3]; CANDIDATES],
    l: usize,
    i: usize,
    n: usize,
) -> [f32; 3] {
    std::array::from_fn(|c| {
        (0..CANDIDATES)
            .map(|k| s.selected[k * 2 * n + l * n + i] * p[k][c])
            .sum()
    })
}
fn prior(
    s: &oracle::Candidates,
    p: &[[f32; 3]; CANDIDATES],
    l: usize,
    i: usize,
    n: usize,
) -> [f32; 3] {
    let total: f32 = (0..CANDIDATES)
        .map(|k| s.prior[k * 2 * n + l * n + i])
        .sum();
    std::array::from_fn(|c| {
        (0..CANDIDATES)
            .map(|k| s.prior[k * 2 * n + l * n + i] * p[k][c])
            .sum::<f32>()
            / total
    })
}
fn write_png(path: &Path, rgb: &[f32], extent: [u32; 2]) -> Result<()> {
    let bytes: Vec<_> = rgb
        .iter()
        .map(|v| {
            let a = v.max(0.0) / (1.0 + v.max(0.0));
            let s = if a < 0.0031308 {
                12.92 * a
            } else {
                1.055 * a.powf(1.0 / 2.4) - 0.055
            };
            (s.clamp(0.0, 1.0) * 255.0).round() as u8
        })
        .collect();
    let mut e = png::Encoder::new(std::fs::File::create(path)?, extent[0], extent[1]);
    e.set_color(png::ColorType::Rgb);
    e.set_depth(png::BitDepth::Eight);
    e.write_header()?.write_image_data(&bytes)?;
    Ok(())
}
#[allow(clippy::needless_range_loop)] // Corresponding stream/frame/lobe/pixel tensors.
fn main() -> Result<()> {
    env_logger::init();
    let mut files = Vec::new();
    let mut out = PathBuf::from("target/noise-risk");
    let mut checkpoint = None::<PathBuf>;
    let mut fit = 3usize;
    let mut args = std::env::args().skip(1);
    while let Some(key) = args.next() {
        if key == "--help" {
            println!(
                "noise-risk --data A.omd --data B.omd ... --fit-realizations N --checkpoint model.safetensors --out DIR\nFirst N streams fit diagnostics; remaining streams are held noise. No trained weights are changed."
            );
            return Ok(());
        }
        let v = args.next().ok_or("missing value")?;
        match key.as_str() {
            "--data" => files.push(PathBuf::from(v)),
            "--out" => out = v.into(),
            "--fit-realizations" => fit = v.parse()?,
            "--checkpoint" => checkpoint = Some(v.into()),
            _ => return Err(format!("unknown option {key}").into()),
        }
    }
    if fit < 2 || files.len() < fit + 2 {
        return Err("need >=2 fitting and >=2 held realizations".into());
    }
    let checkpoint = checkpoint.ok_or("need checkpoint")?;
    let c: Config = ron::from_str(&std::fs::read_to_string(
        checkpoint.with_extension("transport.ron"),
    )?)?;
    let captures = files
        .iter()
        .map(|p| load(p, c))
        .collect::<Result<Vec<_>>>()?;
    validate(&captures)?;
    if out.exists() {
        return Err("refusing to overwrite noise study output directory".into());
    }
    std::fs::create_dir_all(&out)?;
    let context = ommatidia::gpu::create_context(None, false);
    let device = context.device_information().device_name.clone();
    let low = captures[0].frames[0].0.low;
    let mut summaries = Vec::new();
    for causal in [false, true] {
        let mode = if causal { "causal" } else { "reset" };
        let dir = out.join(mode);
        std::fs::create_dir(&dir)?;
        let mut snapshots = Vec::new();
        let mut native_images = Vec::new();
        let mut native_parity = 0.0f64;
        let mut max_dual_gap = 0.0f64;
        let mut cross_noise_fallbacks = [0usize; 2];
        for capture in &captures {
            let mut runtime = native::Native::new(Arc::clone(&context), c, low)?;
            runtime.session.load_checkpoint(&checkpoint)?;
            let mut frames = Vec::new();
            let mut native_frames = Vec::new();
            for (i, (f, _)) in capture.frames.iter().enumerate() {
                if !causal || i % capture.length == 0 {
                    runtime.reset();
                }
                native_frames.push(runtime.process(f)?);
                frames.push(runtime.read_candidates());
            }
            snapshots.push(frames);
            native_images.push(native_frames);
        }
        let mut regressions = vec![noise::Regression::default(); 2 * CANDIDATES];
        // Fit only designated noisy observations. Targets identify risk, never features.
        for r in 0..fit {
            for (f, (frame, target)) in captures[r].frames.iter().enumerate() {
                let n = frame.surfaces.len();
                let s = &snapshots[r][f];
                for l in 0..2 {
                    for i in 0..n {
                        let p = points(s, l, i, n);
                        let truth = std::array::from_fn(|ch| target.lobes[(3 * l + ch) * n + i]);
                        for k in 0..CANDIDATES {
                            if s.prior[k * 2 * n + l * n + i] > 0.0 {
                                regressions[l * CANDIDATES + k]
                                    .add(noise::features(&p, k), noise::mse(p[k], truth))?;
                            }
                        }
                    }
                }
            }
        }
        let weights: Vec<_> = regressions
            .iter()
            .map(|r| {
                if r.samples() == 0 {
                    Ok(None)
                } else {
                    r.fit(1e-3).map(Some)
                }
            })
            .collect::<std::result::Result<_, String>>()?;
        std::fs::write(
            dir.join("risk-model.json"),
            serde_json::to_vec_pretty(&weights)?,
        )?;
        let mut errors = [[0.0f64; METHODS.len()]; 2];
        let mut signed = [[0.0f64; 3]; 2];
        let mut counts = [0usize; 2];
        let mut agree = [0usize; 2];
        let mut comparisons = [0usize; 2];
        let mut risk_by_scale = vec![[0.0f64; 3]; 2 * SCALES];
        let mut scale_count = [0usize; 2];
        let mut per_frame = Vec::new();
        for (f, (frame, target)) in captures[0].frames.iter().enumerate() {
            let n = frame.surfaces.len();
            let mut predictions =
                vec![vec![vec![0.0f32; 6 * n]; METHODS.len()]; captures.len() - fit];
            for l in 0..2 {
                for i in 0..n {
                    let ps: Vec<_> = snapshots.iter().map(|r| points(&r[f], l, i, n)).collect();
                    let truth = std::array::from_fn(|ch| target.lobes[(3 * l + ch) * n + i]);
                    let available = std::array::from_fn(|k| {
                        snapshots[..fit]
                            .iter()
                            .all(|r| r[f].prior[k * 2 * n + l * n + i] > 0.0)
                    });
                    let (shared_choice, _) = noise::transfer(&ps, available, truth, fit)?;
                    let mut winners = Vec::new();
                    for r in 0..captures.len() {
                        winners.push(
                            (0..CANDIDATES)
                                .filter(|k| snapshots[r][f].prior[k * 2 * n + l * n + i] > 0.0)
                                .min_by(|a, b| {
                                    noise::mse(ps[r][*a], truth)
                                        .total_cmp(&noise::mse(ps[r][*b], truth))
                                })
                                .unwrap(),
                        );
                    }
                    let risks: Vec<_> = (0..CANDIDATES)
                        .filter(|k| available[*k])
                        .map(|k| ps.iter().map(|p| noise::mse(p[k], truth)).sum::<f64>())
                        .collect();
                    if risks.iter().copied().fold(0.0, f64::max)
                        - risks.iter().copied().fold(f64::INFINITY, f64::min)
                        > 1e-8
                    {
                        for r in 0..fit {
                            for h in fit..captures.len() {
                                agree[l] += usize::from(winners[r] == winners[h]);
                                comparisons[l] += 1;
                            }
                        }
                    }
                    for k in 0..SCALES {
                        let mean: [f64; 3] = std::array::from_fn(|ch| {
                            ps.iter().map(|p| p[k][ch] as f64).sum::<f64>() / ps.len() as f64
                        });
                        let bias = (0..3)
                            .map(|ch| (mean[ch] - truth[ch] as f64).powi(2))
                            .sum::<f64>()
                            / 3.0;
                        let variance = ps
                            .iter()
                            .map(|p| {
                                (0..3)
                                    .map(|ch| (p[k][ch] as f64 - mean[ch]).powi(2))
                                    .sum::<f64>()
                                    / 3.0
                            })
                            .sum::<f64>()
                            / ps.len() as f64;
                        let row = &mut risk_by_scale[l * SCALES + k];
                        row[0] += bias;
                        row[1] += variance;
                        row[2] += ps.iter().map(|p| noise::mse(p[k], truth)).sum::<f64>()
                            / ps.len() as f64;
                    }
                    scale_count[l] += 1;
                    for r in fit..captures.len() {
                        let s = &snapshots[r][f];
                        let p = &ps[r];
                        let ids: Vec<_> = (0..CANDIDATES)
                            .filter(|k| s.prior[k * 2 * n + l * n + i] > 0.0)
                            .collect();
                        let optimum =
                            oracle::project(&ids.iter().map(|k| p[*k]).collect::<Vec<_>>(), truth)?;
                        max_dual_gap = max_dual_gap.max(optimum.dual_gap);
                        let predicted_choice = ids
                            .iter()
                            .copied()
                            .min_by(|a, b| {
                                let score = |k: usize| {
                                    weights[l * CANDIDATES + k]
                                        .as_ref()
                                        .map_or(f64::INFINITY, |w| {
                                            noise::predict(w, &noise::features(p, k))
                                        })
                                };
                                score(*a).total_cmp(&score(*b))
                            })
                            .unwrap();
                        let shared = if s.prior[shared_choice * 2 * n + l * n + i] > 0.0 {
                            p[shared_choice]
                        } else {
                            cross_noise_fallbacks[l] += 1;
                            prior(s, p, l, i, n)
                        };
                        let images = [
                            selected(s, p, l, i, n),
                            prior(s, p, l, i, n),
                            p[predicted_choice],
                            shared,
                            p[winners[r]],
                            optimum.point,
                        ];
                        for (m, rgb) in images.iter().enumerate() {
                            errors[l][m] += noise::mse(*rgb, truth);
                            for ch in 0..3 {
                                predictions[r - fit][m][(3 * l + ch) * n + i] = rgb[ch];
                            }
                        }
                        for ch in 0..3 {
                            signed[l][ch] += images[0][ch] as f64 - truth[ch] as f64;
                        }
                        counts[l] += 1;
                    }
                }
            }
            for (r, images) in predictions.iter().enumerate() {
                let extent = frame.low.map(|x| x * c.scale);
                let w = extent[0] as usize;
                for (m, lobes) in images.iter().enumerate() {
                    let mut rgb = vec![0.0; 3 * n];
                    for i in 0..n {
                        for ch in 0..3 {
                            rgb[3 * i + ch] = lobes[c.index(low, ch, i % w, i / w)]
                                * frame.surfaces[i].albedo_roughness[ch]
                                + lobes[c.index(low, 3 + ch, i % w, i / w)]
                                + frame.surfaces[i].emission[ch];
                        }
                    }
                    if m == 0 {
                        for (a, b) in rgb.iter().zip(&native_images[r + fit][f]) {
                            native_parity = native_parity
                                .max((*a as f64 - *b as f64).abs() / (1.0 + (*b as f64).abs()));
                        }
                        if native_parity > 1e-5 {
                            return Err(format!("candidate recomposition differs from native output: {native_parity}").into());
                        }
                    }
                    if rgb.iter().any(|v| !v.is_finite()) {
                        return Err("nonfinite remodulated prediction".into());
                    }
                    let linear_mse = rgb
                        .iter()
                        .zip(&target.rgb)
                        .map(|(a, b)| (*a as f64 - *b as f64).powi(2))
                        .sum::<f64>()
                        / rgb.len() as f64;
                    let low_frequency_psnr = -10.0
                        * ommatidia::metrics::low_frequency_error(
                            &rgb,
                            &target.rgb,
                            extent[0] as usize,
                            extent[1] as usize,
                            8,
                        )
                        .max(1e-20)
                        .log10();
                    let detail_ratio =
                        ommatidia::metrics::detail(&rgb, extent[0] as usize, extent[1] as usize)
                            / ommatidia::metrics::detail(
                                &target.rgb,
                                extent[0] as usize,
                                extent[1] as usize,
                            )
                            .max(1e-12);
                    let name = METHODS[m];
                    let psnr = -10.0
                        * (ommatidia::metrics::error(&rgb, &target.rgb) as f64)
                            .max(1e-20)
                            .log10();
                    let energy = rgb.iter().map(|v| *v as f64).sum::<f64>()
                        / target.rgb.iter().map(|v| *v as f64).sum::<f64>().max(1e-12);
                    per_frame.push(serde_json::json!({"stream":r+fit,"frame":f,"method":name,"psnr":psnr,"energy_ratio":energy,"linear_mse":linear_mse,"low_frequency_psnr":low_frequency_psnr,"detail_ratio":detail_ratio}));
                    write_png(
                        &dir.join(format!("{f:03}-noise{}-{name}.png", r + fit)),
                        &rgb,
                        extent,
                    )?;
                }
            }
            write_png(
                &dir.join(format!("{f:03}-reference.png")),
                &target.rgb,
                frame.low.map(|x| x * c.scale),
            )?;
        }
        let mut methods = serde_json::Map::new();
        for (m, name) in METHODS.iter().enumerate() {
            methods.insert(
                (*name).into(),
                serde_json::json!(std::array::from_fn::<_, 2, _>(
                    |l| errors[l][m] / counts[l].max(1) as f64
                )),
            );
        }
        for (l, rows) in risk_by_scale.chunks_exact_mut(SCALES).enumerate() {
            for row in rows {
                row.iter_mut()
                    .for_each(|v| *v /= scale_count[l].max(1) as f64);
            }
        }
        let result = serde_json::json!({"mode":mode,"lobe_mse":methods,"learned_signed_rgb_bias":std::array::from_fn::<_,2,_>(|l|signed[l].map(|v|v/counts[l].max(1) as f64)),
            "winner_agreement":std::array::from_fn::<_,2,_>(|l|agree[l] as f64/comparisons[l].max(1) as f64),"winner_comparisons":comparisons,
            "native_recomposition_max_relative_difference":native_parity,"max_oracle_dual_gap":max_dual_gap,"cross_noise_unavailable_fallbacks":cross_noise_fallbacks,"spatial_bias2_population_variance_mse":risk_by_scale,"frames":per_frame});
        std::fs::write(dir.join("report.json"), serde_json::to_vec_pretty(&result)?)?;
        summaries.push(result);
    }
    std::fs::write(
        out.join("quality.json"),
        serde_json::to_vec_pretty(
            &serde_json::json!({"backend":device,"role":"construction cross-noise diagnostic, not unseen-scene generalization","fit_realizations":fit,"held_realizations":captures.len()-fit,"checkpoint":checkpoint,"files":files,"ridge":1e-3,"risk_target":"linear_lobe_mse",
        "limits":"cross-noise-risk uses true scene reference in fitting streams; oracle rows use held targets and are not deployable; observable-risk has no pixel/scene identifiers; causal streams retain their own learned histories; unobserved noise seeds only, not unseen geometry","results":summaries}),
        )?,
    )?;
    Ok(())
}
