//! Offline image-conditioned field experiment; no blade-volume integration.
use ommatidia::{
    dataset::{Plane, Reader},
    field::{
        self, Config, Manifest, Observations, View,
        data::{self, Prepared, RenderShape, Targets},
        graph, incident, surface,
    },
    rng::Rng,
};
use std::{
    io::Write,
    path::{Path, PathBuf},
    sync::Arc,
};
type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
#[path = "field/diagnostics.rs"]
mod diagnostics;
struct TargetView {
    view: View,
    surface: Option<surface::Capture>,
}
impl std::ops::Deref for TargetView {
    type Target = View;
    fn deref(&self) -> &View {
        &self.view
    }
}
struct Example {
    observations: Observations,
    train: Vec<TargetView>,
    held: TargetView,
    record: field::SceneRecord,
    context_surfaces: Vec<Option<surface::Capture>>,
}
fn load(path: &Path, c: &Config) -> Result<Vec<Example>> {
    let manifest: Manifest =
        serde_json::from_slice(&std::fs::read(path.with_extension("scene.json"))?)?;
    let mut reader = Reader::open(path)?;
    manifest.validate(reader.len())?;
    let layout = *reader.layout();
    if manifest.extent != c.extent || [layout.hr_width(), layout.hr_height()] != c.extent {
        return Err(
            "capture/config extent mismatch; choose --image to match the HR capture".into(),
        );
    }
    let mut examples = Vec::new();
    for (scene, record) in manifest.scenes.into_iter().enumerate() {
        let records: Vec<_> = manifest
            .records
            .iter()
            .filter(|r| r.scene == scene)
            .collect();
        if records.len() < c.views + 2 {
            return Err("need context views, a fitting camera and a held camera per scene".into());
        }
        let mut views = Vec::new();
        for r in records {
            let sample = reader.sample(r.sample)?;
            let n = layout.hr_texels();
            let mut rgb = vec![0.0; 3 * n];
            for channel in 0..3 {
                let plane = sample
                    .hr_channel(&layout, Plane::Color, channel)
                    .ok_or("missing HR RGB")?;
                for i in 0..n {
                    rgb[3 * i + channel] = plane[i].to_f32();
                }
            }
            if rgb.iter().any(|v| !v.is_finite() || *v < 0.0) {
                return Err("invalid scene-linear RGB target".into());
            }
            // No other Sample plane is read by this path.
            views.push(TargetView {
                view: View {
                    camera: r.camera,
                    rgb,
                },
                surface: r.surface.clone(),
            });
        }
        let held = views.pop().unwrap();
        let n = views.len();
        let context_indices: Vec<_> = (0..c.views).map(|v| v * n / c.views).collect();
        let context: Vec<_> = context_indices
            .iter()
            .map(|i| views[*i].view.clone())
            .collect();
        let context_surfaces = context_indices
            .iter()
            .map(|i| views[*i].surface.clone())
            .collect();
        let train: Vec<_> = views
            .into_iter()
            .enumerate()
            .filter(|(i, _)| !context_indices.contains(i))
            .map(|(_, v)| v)
            .collect();
        if context
            .iter()
            .any(|v| v.camera.origin == held.camera.origin)
            || train.is_empty()
        {
            return Err(
                "held camera duplicates context or no fitting views remain; use --field-views"
                    .into(),
            );
        }
        let observations = Observations {
            bounds: record.bounds,
            views: context,
        };
        observations.validate(c)?;
        examples.push(Example {
            observations,
            train,
            held,
            record,
            context_surfaces,
        });
    }
    Ok(examples)
}
fn copy(source: &meganeura::Session, dest: &mut meganeura::Session, model: &graph::Network) {
    for p in &model.params {
        let mut data = vec![0.0; p.len];
        source.read_param(&p.name, &mut data);
        dest.set_parameter(&p.name, &data);
    }
}
/// Labels specify evaluator rays, never inference observations. No held score
/// participates in optimization or checkpoint selection.
fn score_incident(
    session: &mut meganeura::Session,
    c: &Config,
    shape: RenderShape,
    example: &Example,
) -> Result<serde_json::Value> {
    let Some(capture) = &example.record.incident else {
        return Ok(serde_json::Value::Null);
    };
    let mut direct = Vec::new();
    let mut indirect = Vec::new();
    for batch in capture.probes.chunks(shape.rays) {
        let rays: Vec<_> = (0..shape.rays)
            .map(|i| batch[i.min(batch.len() - 1)].ray())
            .collect();
        let (q, dt) = data::ray_queries(example.observations.bounds, &rays, shape.steps)?;
        Prepared::new(&example.observations, c, &q)?.feed(session);
        session.set_input("ray.deltas", &dt);
        session.step();
        session.wait();
        let mut d = vec![0.0; shape.rays * 3];
        let mut b = d.clone();
        session.read_output_by_index(1, &mut d);
        session.read_output_by_index(2, &mut b);
        if d.iter().chain(&b).any(|v| !v.is_finite() || *v < 0.0) {
            return Err("invalid incident prediction".into());
        }
        direct.extend_from_slice(&d[..3 * batch.len()]);
        indirect.extend_from_slice(&b[..3 * batch.len()]);
    }
    let total: Vec<_> = direct.iter().zip(&indirect).map(|(a, b)| a + b).collect();
    let truth_d: Vec<_> = capture.probes.iter().flat_map(|p| p.direct.mean).collect();
    let truth_i: Vec<_> = capture
        .probes
        .iter()
        .flat_map(|p| p.indirect.mean)
        .collect();
    let truth_t: Vec<_> = capture.probes.iter().flat_map(|p| p.total()).collect();
    let mean_variance = capture
        .probes
        .iter()
        .flat_map(|p| p.total_variance_of_mean)
        .map(|v| v as f64)
        .sum::<f64>()
        / truth_t.len() as f64;
    Ok(
        serde_json::json!({"probes":capture.probes.len(),"direct":score(&direct,&truth_d),
        "indirect":score(&indirect,&truth_i),"total":score(&total,&truth_t),
        "target_mean_variance":mean_variance,"target_paths_per_probe":capture.paths_per_batch*capture.batches}),
    )
}
fn score(a: &[f32], b: &[f32]) -> serde_json::Value {
    let log_mse = a
        .iter()
        .zip(b)
        .map(|(a, b)| (a.ln_1p() - b.ln_1p()).powi(2) as f64)
        .sum::<f64>()
        / a.len() as f64;
    let mse = a
        .iter()
        .zip(b)
        .map(|(a, b)| (a - b).powi(2) as f64)
        .sum::<f64>()
        / a.len() as f64;
    serde_json::json!({"log1p_mse":log_mse,"linear_mse":mse,"compressed_psnr":-10.0*(ommatidia::metrics::error(a,b) as f64).max(1e-20).log10()})
}
fn png(path: &Path, rgb: &[f32], extent: [u32; 2]) -> Result<()> {
    let bytes: Vec<_> = rgb
        .iter()
        .map(|v| {
            let x = v.max(0.0) / (1.0 + v.max(0.0));
            let x = if x <= 0.0031308 {
                12.92 * x
            } else {
                1.055 * x.powf(1.0 / 2.4) - 0.055
            };
            (255.0 * x.clamp(0.0, 1.0)).round() as u8
        })
        .collect();
    let mut e = png::Encoder::new(std::fs::File::create(path)?, extent[0], extent[1]);
    e.set_color(png::ColorType::Rgb);
    e.set_depth(png::BitDepth::Eight);
    e.write_header()?.write_image_data(&bytes)?;
    Ok(())
}
fn main() -> Result<()> {
    env_logger::init();
    let mut c = Config::default();
    let mut data_files = Vec::<PathBuf>::new();
    let mut eval = None;
    let mut out = PathBuf::from("target/field");
    let mut steps = 256;
    let mut seed = 7u64;
    let mut shape = RenderShape {
        rays: 16,
        steps: 32,
        probes: 16,
    };
    let mut rate = 1e-3;
    let mut incident_rays = 0usize;
    let mut incident_weight = 0.1f32;
    let mut surface_weight = None::<f32>;
    let mut visibility_weight = 0.05f32;
    let mut consistency_rays = 0usize;
    let mut consistency_weight = 0.05f32;
    let mut emitter_fraction = 0.0f32;
    let mut diagnostics = false;
    let mut stratified = false;
    let mut eval_checkpoint = None::<PathBuf>;
    let mut argv = std::env::args().skip(1);
    while let Some(flag) = argv.next() {
        if flag == "--stratified" {
            stratified = true;
            continue;
        }
        if flag == "--diagnostics" {
            diagnostics = true;
            continue;
        }
        if flag == "--help" || flag == "-h" {
            println!(
                "field --data CAPTURE.omd [--data OTHER.omd] [--eval-data UNSEEN.omd]\n  --out DIR --steps N --seed N --image N --views N --channels N --hidden N\n  --rays N --samples N --probes N --rate F --stratified\n  --view-fusion moments|late-rgb|visible-rgb --visibility-weight F [0.05] --eval-checkpoint PATH\n  --surface-weight F (opt-in; 0 retains matched control graph)\n  --consistency-rays N [0] --consistency-weight F [0.05]\n  --emitter-fraction F [0] --diagnostics\n  --incident-rays N [0] --incident-weight F [0.1 when incident rays enabled]\nPosed RGB only. Final camera is held; light labels supervise separate heads.\nOutput: weights/config, RGB contexts, fixed-budget held-camera quality, PNGs."
            );
            return Ok(());
        }
        let v = argv.next().ok_or(format!("{flag} requires a value"))?;
        match flag.as_str() {
            "--data" => data_files.push(v.into()),
            "--eval-data" => eval = Some(PathBuf::from(v)),
            "--eval-checkpoint" => eval_checkpoint = Some(PathBuf::from(v)),
            "--visibility-weight" => visibility_weight = v.parse()?,
            "--consistency-rays" => consistency_rays = v.parse()?,
            "--consistency-weight" => consistency_weight = v.parse()?,
            "--view-fusion" => {
                c.view_fusion = match v.as_str() {
                    "moments" => field::ViewFusion::Moments,
                    "late-rgb" => field::ViewFusion::LateRgb,
                    "visible-rgb" => field::ViewFusion::VisibleRgb,
                    _ => return Err("view fusion must be moments, late-rgb or visible-rgb".into()),
                }
            }
            "--out" => out = v.into(),
            "--steps" => steps = v.parse()?,
            "--seed" => seed = v.parse()?,
            "--image" => {
                let n = v.parse()?;
                c.extent = [n, n];
            }
            "--views" => c.views = v.parse()?,
            "--channels" => c.channels = v.parse()?,
            "--hidden" => c.hidden = v.parse()?,
            "--rays" => shape.rays = v.parse()?,
            "--samples" => shape.steps = v.parse()?,
            "--probes" => shape.probes = v.parse()?,
            "--rate" => rate = v.parse()?,
            "--incident-rays" => incident_rays = v.parse()?,
            "--incident-weight" => incident_weight = v.parse()?,
            "--surface-weight" => surface_weight = Some(v.parse()?),
            "--emitter-fraction" => emitter_fraction = v.parse()?,
            _ => return Err(format!("unknown option {flag}").into()),
        }
    }
    if surface_weight.is_some_and(|w| !w.is_finite() || w < 0.0)
        || !emitter_fraction.is_finite()
        || !(0.0..=1.0).contains(&emitter_fraction)
    {
        return Err("surface weight must be nonnegative; emitter fraction must be in [0,1]".into());
    }
    if let Some(path) = &eval_checkpoint {
        let source = path.canonicalize()?;
        let directory = source
            .parent()
            .ok_or("checkpoint has no parent directory")?;
        if out.exists() && out.canonicalize()? == directory {
            return Err("checkpoint recovery requires a separate output directory".into());
        }
        c = serde_json::from_slice(&std::fs::read(path.with_file_name("model.field.json"))?)?;
        steps = 0;
    }
    c.validate()?;
    shape.queries()?;
    let training_shape = RenderShape {
        rays: shape
            .rays
            .checked_add(incident_rays)
            .and_then(|n| n.checked_add(consistency_rays))
            .ok_or("too many rays")?,
        ..shape
    };
    training_shape.queries()?;
    if !consistency_weight.is_finite()
        || consistency_weight < 0.0
        || (consistency_rays != 0
            && (c.view_fusion != field::ViewFusion::VisibleRgb
                || !shape.steps.is_multiple_of(field::visibility::BINS)
                || eval_checkpoint.is_some()))
    {
        return Err("consistency needs visible-rgb training, samples divisible by 16, and finite nonnegative weight".into());
    }
    if !incident_weight.is_finite() || incident_weight < 0.0 {
        return Err("incident weight must be finite and nonnegative".into());
    }
    if incident_rays == 0 {
        incident_weight = 0.0;
    }
    if data_files.is_empty()
        || (steps == 0 && eval_checkpoint.is_none())
        || !f32::is_finite(rate)
        || rate <= 0.0
    {
        return Err("need data, positive updates and finite positive rate".into());
    }
    let mut train = Vec::new();
    for path in &data_files {
        train.extend(load(path, &c)?);
    }
    if train.is_empty() {
        return Err("empty corpus".into());
    }
    if incident_rays > 0 && train.iter().any(|e| e.record.incident.is_none()) {
        return Err(
            "incident training requires --incident-probes captures for every fitting scene".into(),
        );
    }
    if (surface_weight.is_some() || emitter_fraction > 0.0)
        && train
            .iter()
            .any(|e| e.train.iter().any(|v| v.surface.is_none()))
    {
        return Err(
            "surface training requires --surface-labels captures, including the weight-zero arm"
                .into(),
        );
    }
    if !visibility_weight.is_finite() || visibility_weight < 0.0 {
        return Err("visibility weight must be finite and nonnegative".into());
    }
    let visibility_targets =
        if c.view_fusion == field::ViewFusion::VisibleRgb && eval_checkpoint.is_none() {
            train
                .iter()
                .map(|e| {
                    field::visibility::Targets::new(
                        &e.observations,
                        &c,
                        &e.context_surfaces,
                        visibility_weight,
                    )
                })
                .collect::<std::result::Result<Vec<_>, _>>()?
        } else {
            Vec::new()
        };
    let held = eval.as_ref().map(|p| load(p, &c)).transpose()?;
    if held.as_ref().is_some_and(|held| {
        held.iter().any(|a| {
            train
                .iter()
                .any(|b| a.record.scene_seed == b.record.scene_seed)
        })
    }) {
        return Err(
            "--eval-data must contain unseen scene seeds, including across lighting variants"
                .into(),
        );
    }
    std::fs::create_dir_all(&out)?;
    let context = ommatidia::gpu::create_context(None, false);
    let backend = context.device_information().device_name.clone();
    println!("field backend {backend}; quality only");
    let model = if eval_checkpoint.is_some() {
        graph::build_diagnostics(&c, shape)?
    } else if consistency_rays != 0 {
        graph::build_consistent_training(
            &c,
            training_shape,
            incident_rays,
            consistency_rays,
            surface_weight.is_some(),
        )?
    } else if surface_weight.is_some() {
        graph::build_surface_training(&c, training_shape, incident_rays)?
    } else {
        graph::build_training(&c, training_shape, incident_rays)?
    };
    let mut session = if eval_checkpoint.is_some() {
        ommatidia::gpu::inference_session(&model.graph, Arc::clone(&context))
    } else {
        ommatidia::gpu::training_session(&model.graph, Arc::clone(&context))
    };
    model.initialize(&mut session, seed);
    let inference = graph::build_diagnostics(&c, shape)?;
    let mut baseline = ommatidia::gpu::inference_session(&inference.graph, Arc::clone(&context));
    copy(&session, &mut baseline, &inference);
    let mut rng = Rng::new(seed);
    let mut first = 0.0;
    let mut last = 0.0;
    let mut log = std::fs::File::create(out.join("loss.csv"))?;
    writeln!(log, "step,loss")?;
    for step in 0..steps {
        let example = &train[step % train.len()];
        let target = &example.train[rng.below(example.train.len() as u32) as usize];
        let n = (c.extent[0] * c.extent[1]) as usize;
        // Balanced source pixels are an explicit training stratum, shared across
        // ablations. They never change ray depth samples or inference observations.
        let source_pixels: Vec<_> = target
            .surface
            .as_ref()
            .map(|s| {
                s.emission
                    .iter()
                    .enumerate()
                    .filter(|(_, e)| e.iter().any(|v| *v > 0.0))
                    .map(|(i, _)| i)
                    .collect()
            })
            .unwrap_or_default();
        let pixels: Vec<_> = (0..shape.rays)
            .map(|i| {
                if !source_pixels.is_empty() && (i as f32) < emitter_fraction * shape.rays as f32 {
                    source_pixels[rng.below(source_pixels.len() as u32) as usize]
                } else {
                    rng.below(n as u32) as usize
                }
            })
            .collect();
        let mut rays: Vec<_> = pixels
            .iter()
            .map(|p| {
                target.camera.ray(
                    [
                        (p % c.extent[0] as usize) as f32,
                        (p / c.extent[0] as usize) as f32,
                    ],
                    c.extent,
                )
            })
            .collect();
        // Separate RNG keeps camera pixels and emission probes identical in
        // incident-loss ablations, including the weight-zero arm.
        let mut probe_rng = Rng::new(seed ^ (step as u64).wrapping_mul(0xD1B5_4A32_D192_ED03));
        let chosen: Vec<_> = (0..incident_rays)
            .map(|_| {
                let probes = &example.record.incident.as_ref().unwrap().probes;
                &probes[probe_rng.below(probes.len() as u32) as usize]
            })
            .collect();
        rays.extend(chosen.iter().map(|p| p.ray()));
        let source_batch = if consistency_rays != 0 {
            Some(field::consistency::Batch::new(
                &example.observations,
                &c,
                consistency_rays,
                seed ^ 0xBE54_66CF_34E9_0C6C ^ (step as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15),
            )?)
        } else {
            None
        };
        if let Some(batch) = &source_batch {
            rays.extend_from_slice(&batch.rays);
        }
        // Independent stream: changing sample count/mode must not change
        // fitting camera pixels or emission/incident-probe choices.
        let sample_seed =
            seed ^ 0xA24B_AED4_963E_E407 ^ (step as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        let (mut queries, deltas) = if stratified {
            data::stratified_ray_queries(
                example.observations.bounds,
                &rays,
                shape.steps,
                sample_seed,
            )?
        } else {
            data::ray_queries(example.observations.bounds, &rays, shape.steps)?
        };
        let (emission, emission_mask) = data::append_probes(
            &mut queries,
            &example.record.lighting,
            shape.probes,
            seed + step as u64,
        );
        let targets = Targets {
            rgb: pixels
                .iter()
                .flat_map(|p| target.rgb[3 * p..3 * p + 3].iter().copied())
                .collect(),
            emission,
            emission_mask,
            environment: example.record.lighting.environment,
            environment_mask: [1.0; 3],
        };
        Prepared::new(&example.observations, &c, &queries)?.feed(&mut session);
        session.set_input("ray.deltas", &deltas);
        targets.feed(&mut session);
        if let Some(batch) = &source_batch {
            batch.feed(&mut session, consistency_weight)?;
        }
        if !visibility_targets.is_empty() {
            visibility_targets[step % train.len()].feed(&mut session);
        }
        if let Some(weight) = surface_weight {
            let labels = target.surface.as_ref().unwrap();
            let mut chosen: Vec<_> = pixels
                .iter()
                .map(|p| Some((labels.distance[*p], labels.ray_limit)))
                .collect();
            chosen.extend((0..incident_rays + consistency_rays).map(|_| None));
            let termination = surface::Targets::new(
                example.observations.bounds,
                &rays,
                shape.steps,
                &chosen,
                weight,
            )?;
            termination.feed(&mut session);
        }
        if !chosen.is_empty() {
            incident::Targets::new(&chosen, c.exposure)?
                .weighted(incident_weight)?
                .feed(&mut session);
        }
        session.set_adam(rate, 0.9, 0.999, 1e-8);
        session.step();
        session.wait();
        last = session.read_loss();
        if !last.is_finite() {
            return Err(format!("non-finite loss at update {step}").into());
        }
        if step == 0 {
            first = last;
        }
        writeln!(log, "{step},{last}")?;
        if step % 16 == 0 {
            println!("field update {step}/{steps}: {last:.6}");
        }
    }
    if eval_checkpoint.is_none() {
        session.save_checkpoint(&out.join("model.safetensors"))?;
    }
    let checkpoint = eval_checkpoint
        .clone()
        .unwrap_or_else(|| out.join("model.safetensors"));
    std::fs::write(out.join("model.field.json"), serde_json::to_vec_pretty(&c)?)?;
    // Reload before any held-camera evaluation. No metric-based checkpoint selection.
    let mut learned = ommatidia::gpu::inference_session(&inference.graph, Arc::clone(&context));
    learned.load_checkpoint(&checkpoint)?;
    let incident_shape = RenderShape {
        rays: shape.rays,
        steps: shape.steps,
        probes: 0,
    };
    let incident_model = graph::build_incident(&c, incident_shape)?;
    let mut incident_session =
        ommatidia::gpu::inference_session(&incident_model.graph, Arc::clone(&context));
    incident_session.load_checkpoint(&checkpoint)?;
    let examples = held.as_ref().unwrap_or(&train);
    let mut scores = Vec::new();
    for (i, example) in examples.iter().enumerate() {
        let initial = diagnostics::render(
            &mut baseline,
            &c,
            shape,
            &example.observations,
            &example.held,
        )?;
        let prediction = diagnostics::render(
            &mut learned,
            &c,
            shape,
            &example.observations,
            &example.held,
        )?;
        let mut mean = [0.0f32; 3];
        let mut count = 0;
        for view in &example.observations.views {
            for p in view.rgb.chunks_exact(3) {
                for c in 0..3 {
                    mean[c] += p[c];
                }
                count += 1;
            }
        }
        mean.iter_mut().for_each(|v| *v /= count as f32);
        let n = example.held.rgb.len() / 3;
        let constant: Vec<_> = (0..n).flat_map(|_| mean).collect();
        png(
            &out.join(format!("{i}-reference.png")),
            &example.held.rgb,
            c.extent,
        )?;
        png(
            &out.join(format!("{i}-prediction.png")),
            &prediction.rgb,
            c.extent,
        )?;
        png(
            &out.join(format!("{i}-initial.png")),
            &initial.rgb,
            c.extent,
        )?;
        // Reusable inference asset: RGB and poses, explicitly no training lights.
        std::fs::write(
            out.join(format!("{i}-context.json")),
            serde_json::to_vec(&example.observations)?,
        )?;
        let mut diagnostic = serde_json::Value::Null;
        if diagnostics {
            let fit = diagnostics::render(
                &mut learned,
                &c,
                shape,
                &example.observations,
                &example.train[0],
            )?;
            png(&out.join(format!("{i}-fit.png")), &fit.rgb, c.extent)?;
            png(
                &out.join(format!("{i}-fit-reference.png")),
                &example.train[0].rgb,
                c.extent,
            )?;
            diagnostics::save_geometry(
                &out.join(format!("{i}-held")),
                &prediction,
                &example.observations,
                &example.held,
                &c,
                shape.steps,
            )?;
            diagnostics::save_geometry(
                &out.join(format!("{i}-fit")),
                &fit,
                &example.observations,
                &example.train[0],
                &c,
                shape.steps,
            )?;
            let mut controls = Vec::new();
            for zero in [true, false] {
                let obs = diagnostics::ablated(&example.observations, zero);
                let control = diagnostics::render(&mut learned, &c, shape, &obs, &example.held)?;
                let name = if zero {
                    "zero-rgb"
                } else {
                    "scrambled-rgb-poses"
                };
                png(&out.join(format!("{i}-{name}.png")), &control.rgb, c.extent)?;
                controls.push(
                    serde_json::json!({"input":name,"quality":score(&control.rgb,&example.held.rgb),
                    "output_change":score(&control.rgb,&prediction.rgb)}),
                );
            }
            diagnostic = serde_json::json!({"fitting_camera":score(&fit.rgb,&example.train[0].rgb),
                "geometry_fit":diagnostics::geometry(&fit,&example.observations,&example.train[0],&c,shape.steps),
                "geometry_held":diagnostics::geometry(&prediction,&example.observations,&example.held,&c,shape.steps),
                "context_controls":controls});
        }
        scores.push(serde_json::json!({"scene_seed":example.record.scene_seed,"learned":score(&prediction.rgb,&example.held.rgb),
            "untrained":score(&initial.rgb,&example.held.rgb),"context_mean":score(&constant,&example.held.rgb),"black":score(&vec![0.0;3*n],&example.held.rgb),
            "source_consistency":if diagnostics && c.view_fusion == field::ViewFusion::VisibleRgb {
                diagnostics::consistency(&mut learned,&c,shape,example)?
            } else { serde_json::Value::Null },
            "diagnostics":diagnostic,"incident":score_incident(&mut incident_session,&c,incident_shape,example)?}));
    }
    let report = serde_json::json!({"backend":backend,"quality_only":true,"eval_checkpoint":eval_checkpoint,"view_fusion":c.view_fusion,"steps":steps,"seed":seed,"first_loss":first,"last_loss":last,
        "rays":shape.rays,"samples":shape.steps,"probes":shape.probes,"incident_rays":incident_rays,"incident_weight":incident_weight,"training_files":data_files,
        "surface_weight":surface_weight,"visibility_weight":visibility_weight,
        "consistency_rays":consistency_rays,"consistency_weight":consistency_weight,"emitter_fraction":emitter_fraction,"diagnostics":diagnostics,
        "sampling":if stratified {"stratified-fixed-intervals"} else {"midpoint"},
        "evaluation_sampling":"midpoint","parameter_count":model.params.iter().map(|p|p.len).sum::<usize>(),
        "image_rays_seen":steps as u64 * shape.rays as u64,
        "ray_queries_seen":steps as u64 * training_shape.rays as u64 * shape.steps as u64,
        "evaluation":if held.is_some(){"unseen scenes and held cameras"}else{"held cameras of fitting scenes"},"scores":scores});
    std::fs::write(
        out.join("quality.json"),
        serde_json::to_vec_pretty(&report)?,
    )?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}
