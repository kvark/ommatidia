//! GPU pass-span timing and resident-memory measurement, separate from quality evaluation.
use ommatidia::transport::{Config, Frame, native::Native};
use ommatidia_train::{Result, open_capture};
use std::{io::Write, path::PathBuf, sync::Arc};

fn statistics(values: &[f64]) -> serde_json::Value {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let percentile = |q: f64| sorted[(q * sorted.len() as f64).ceil() as usize - 1];
    serde_json::json!({
        "samples": values.len(), "mean_ms": values.iter().sum::<f64>() / values.len() as f64,
        "p50_ms": percentile(0.5), "p95_ms": percentile(0.95),
        "min_ms": sorted[0], "max_ms": sorted[sorted.len()-1],
    })
}

fn main() -> Result<()> {
    env_logger::init();
    let mut checkpoint = None;
    let mut data = None;
    let mut out = None;
    let mut device = None;
    let mut sequence = 0usize;
    let mut repeats = 3usize;
    let mut warmup = 2usize;
    let mut args = std::env::args().skip(1);
    while let Some(flag) = args.next() {
        if flag == "--help" {
            println!(
                "benchmark --checkpoint FILE --data FILE --out NEW_DIRECTORY [--device-id ID] [--sequence N] [--repeats 3] [--warmup-sequences 2]\nGPU pass-span times only: preparation, ordinary grouped neural execution and resolve. CPU upload, readback, submission/queue gaps and compilation are excluded. Run without concurrent GPU workloads. Memory includes Native's offline buffers and reports driver usage separately."
            );
            return Ok(());
        }
        let value = args.next().ok_or("missing option value")?;
        match flag.as_str() {
            "--checkpoint" => checkpoint = Some(PathBuf::from(value)),
            "--data" => data = Some(PathBuf::from(value)),
            "--out" => out = Some(PathBuf::from(value)),
            "--device-id" => device = Some(ommatidia::gpu::parse_device_id(&value)?),
            "--sequence" => sequence = value.parse()?,
            "--repeats" => repeats = value.parse()?,
            "--warmup-sequences" => warmup = value.parse()?,
            _ => return Err(format!("unknown option {flag}").into()),
        }
    }
    if repeats == 0 || warmup == 0 {
        return Err("positive warmup and measurement repeat counts required".into());
    }
    let checkpoint = checkpoint.ok_or("--checkpoint required")?;
    let data = data.ok_or("--data required")?;
    let out = out.ok_or("--out required")?;
    let config: Config = ron::from_str(&std::fs::read_to_string(
        checkpoint
            .parent()
            .ok_or("checkpoint has no parent")?
            .join("model.transport.ron"),
    )?)?;
    let (mut reader, capture) = open_capture(&data)?;
    let layout = *reader.layout();
    let length = reader.sequence_length();
    if sequence >= reader.len() / length {
        return Err("sequence index out of bounds".into());
    }
    let frames = (sequence * length..(sequence + 1) * length)
        .map(|i| Ok(Frame::from_sample(&reader.sample(i)?, layout, config)?))
        .collect::<Result<Vec<_>>>()?;
    std::fs::create_dir(&out)?;
    let context = ommatidia::gpu::create_context(device, true);
    let mut model = Native::with_timing(Arc::clone(&context), config, frames[0].low, true)?;
    model.session.load_checkpoint(&checkpoint)?;
    let mut max_device_usage = None;
    let mut min_device_budget = None;
    let mut record_memory = |model: &Native| {
        if let Some(stats) = model.session.device_memory_stats() {
            max_device_usage = Some(max_device_usage.unwrap_or(0).max(stats.usage_bytes));
            min_device_budget = Some(
                min_device_budget
                    .unwrap_or(u64::MAX)
                    .min(stats.budget_bytes),
            );
        }
    };
    record_memory(&model);
    let mut rows = std::fs::File::create(out.join("frames.csv"))?;
    writeln!(
        rows,
        "repeat,frame,prepare_ms,network_ms,resolve_ms,total_ms"
    )?;
    let mut durations: [Vec<f64>; 4] = std::array::from_fn(|_| Vec::new());
    for repeat in 0..warmup + repeats {
        model.reset();
        for (frame_index, frame) in frames.iter().enumerate() {
            model.advance(frame)?;
            record_memory(&model);
            let times = model
                .gpu_timings()
                .ok_or("missing GPU timings")?
                .map(|d| d.as_secs_f64() * 1000.0);
            let total = times.iter().sum::<f64>();
            if !total.is_finite() || total <= 0.0 {
                return Err("invalid GPU timestamps".into());
            }
            if repeat >= warmup {
                writeln!(
                    rows,
                    "{},{frame_index},{:.9},{:.9},{:.9},{total:.9}",
                    repeat - warmup,
                    times[0],
                    times[1],
                    times[2]
                )?;
                for (values, value) in durations
                    .iter_mut()
                    .zip([times[0], times[1], times[2], total])
                {
                    values.push(value);
                }
            }
        }
    }
    rows.flush()?;
    let information = context.device_information();
    let report = serde_json::json!({
        "device": {"name": information.device_name, "driver": information.driver_name,
            "driver_info": information.driver_info, "software": information.is_software_emulated},
        "checkpoint": checkpoint, "dataset": data, "capture": capture, "config": config,
        "sequence": sequence, "sequence_frames": length, "warmup_sequences": warmup, "measured_sequences": repeats,
        "input_extent": frames[0].low, "output_extent": frames[0].low.map(|v|v*config.scale),
        "method": "hardware GPU pass-span durations; normal grouped dispatch execution, not per-dispatch profiling",
        "excluded_from_time": "capture, compilation, CPU validation/upload/readback, submission and inter-submission queue gaps; not end-to-end frame time or throughput",
        "percentiles": "nearest rank",
        "prepare": statistics(&durations[0]), "network": statistics(&durations[1]),
        "resolve": statistics(&durations[2]), "total": statistics(&durations[3]),
        "memory": {
            "requested_resident_buffer_bytes": model.buffer_memory_bytes(),
            "max_observed_device_usage_bytes": max_device_usage, "minimum_reported_device_budget_bytes": min_device_budget,
            "scope": "resident buffers include Native offline upload/readback; driver usage sampled after loading and each completed frame, including warmup; not a transient peak or host-RAM measurement; unsupported driver queries are null"
        },
        "validation_enabled": cfg!(debug_assertions),
    });
    std::fs::write(
        out.join("performance.json"),
        serde_json::to_vec_pretty(&report)?,
    )?;
    println!("{}", serde_json::to_string_pretty(&report["total"])?);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn percentiles_use_nearest_rank_without_reordering_samples() {
        let values = [4.0, 1.0, 3.0, 2.0];
        let s = statistics(&values);
        assert_eq!(s["mean_ms"], 2.5);
        assert_eq!(s["p50_ms"], 2.0);
        assert_eq!(s["p95_ms"], 4.0);
        assert_eq!(statistics(&[7.0])["p50_ms"], 7.0);
        assert_eq!(values[0], 4.0);
    }
}
