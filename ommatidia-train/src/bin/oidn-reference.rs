//! Offline OIDN RT quality reference: native LR denoising, then jitter-aware 2x resampling.
use ommatidia::{
    dataset::{Layout, Plane, Sample},
    metrics,
};
use ommatidia_train::{Result, open_capture, save_linear, save_png};
use std::{
    io::{BufRead, Read, Write},
    path::PathBuf,
    process::Command,
};

fn write_pfm(mut file: impl Write, rgb: &[f32], extent: [usize; 2]) -> Result<()> {
    if extent.contains(&0) || rgb.len() != extent[0] * extent[1] * 3 {
        return Err("invalid PFM extent or pixel count".into());
    }
    writeln!(file, "PF\n{} {}\n-1.0", extent[0], extent[1])?;
    for row in rgb.chunks_exact(extent[0] * 3).rev() {
        for value in row {
            file.write_all(&value.to_le_bytes())?;
        }
    }
    file.flush()?;
    Ok(())
}

fn read_pfm(file: impl Read, extent: [usize; 2]) -> Result<Vec<f32>> {
    let mut file = std::io::BufReader::new(file);
    let mut lines = Vec::new();
    for _ in 0..3 {
        let mut line = String::new();
        file.read_line(&mut line)?;
        lines.push(line.trim().to_owned());
    }
    if lines[0] != "PF"
        || lines[1] != format!("{} {}", extent[0], extent[1])
        || lines[2].parse::<f32>()? != -1.0
    {
        return Err("expected little-endian RGB PFM at native input extent".into());
    }
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes)?;
    if bytes.len() != extent[0] * extent[1] * 12 {
        return Err("invalid PFM byte count".into());
    }
    let mut result = Vec::with_capacity(bytes.len() / 4);
    for row in bytes.chunks_exact(extent[0] * 12).rev() {
        result.extend(
            row.chunks_exact(4)
                .map(|v| f32::from_le_bytes(v.try_into().unwrap())),
        );
    }
    if result.iter().any(|v| !v.is_finite() || *v < 0.0) {
        return Err("invalid OIDN radiance".into());
    }
    Ok(result)
}

fn observed(sample: &Sample, layout: Layout) -> Result<([Vec<f32>; 3], [f32; 2])> {
    let n = layout.lr_texels();
    let plane = |plane, c| {
        sample
            .lr_channel(&layout, plane, c)
            .ok_or("missing OIDN observation")
    };
    let mut inputs = [vec![0.0; 3 * n], vec![0.0; 3 * n], vec![0.0; 3 * n]];
    for c in 0..3 {
        let color = plane(Plane::Color, c)?;
        let diffuse = plane(Plane::DiffuseAlbedo, c)?;
        let specular = plane(Plane::SpecularF0, c)?;
        let normal = plane(Plane::Normal, c)?;
        for i in 0..n {
            inputs[0][3 * i + c] = color[i].to_f32();
            inputs[1][3 * i + c] = (diffuse[i].to_f32() + specular[i].to_f32()).clamp(0.0, 1.0);
            inputs[2][3 * i + c] = normal[i].to_f32();
        }
    }
    for normal in inputs[2].chunks_exact_mut(3) {
        let length = normal.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-6);
        for v in normal {
            *v /= length;
        }
    }
    let jitter = [0, 1].map(|c| {
        sample
            .lr_channel(&layout, Plane::Jitter, c)
            .map_or(0.0, |v| v[0].to_f32())
    });
    if inputs.iter().flatten().any(|v| !v.is_finite()) || jitter.iter().any(|v| !v.is_finite()) {
        return Err("non-finite OIDN observation".into());
    }
    Ok((inputs, jitter))
}

fn resample(rgb: &[f32], low: [usize; 2], scale: usize, jitter: [f32; 2]) -> Vec<f32> {
    let high = low.map(|v| v * scale);
    let mut result = vec![0.0; high[0] * high[1] * 3];
    for y in 0..high[1] {
        for x in 0..high[0] {
            let q = [0, 1].map(|axis| {
                ((([x, y][axis] as f32 + 0.5) / scale as f32) - 0.5 - jitter[axis])
                    .clamp(0.0, (low[axis] - 1) as f32)
            });
            let at = q.map(|v| v.floor() as usize);
            let t = [q[0] - at[0] as f32, q[1] - at[1] as f32];
            for k in 0..4 {
                let sx = (at[0] + k % 2).min(low[0] - 1);
                let sy = (at[1] + k / 2).min(low[1] - 1);
                let weight = (if k % 2 == 0 { 1.0 - t[0] } else { t[0] })
                    * (if k / 2 == 0 { 1.0 - t[1] } else { t[1] });
                for c in 0..3 {
                    result[(y * high[0] + x) * 3 + c] += weight * rgb[(sy * low[0] + sx) * 3 + c];
                }
            }
        }
    }
    result
}

fn main() -> Result<()> {
    let mut data = Vec::new();
    let mut out = None;
    let mut executable = None;
    let mut linear = false;
    let mut limit = usize::MAX;
    let mut args = std::env::args().skip(1);
    while let Some(flag) = args.next() {
        match flag.as_str() {
            "--eval-only" => (),
            "--save-linear" => linear = true,
            "--help" => {
                println!(
                    "oidn-reference --oidn PATH_TO_oidnDenoise --eval-data FILE --out DIR [--save-linear] [--limit N]\nRepeat --eval-data in the same order as the learned evaluation. Native LR, HDR/high quality, clean primary guides, CPU/4 threads; bilinear upsample only after denoising. No history, HR guides or target inputs. --eval-only is accepted for the common evaluation protocol."
                );
                return Ok(());
            }
            "--eval-data" => data.push(PathBuf::from(args.next().ok_or("missing dataset")?)),
            "--out" => {
                out = Some(PathBuf::from(
                    args.next().ok_or("missing output directory")?,
                ))
            }
            "--oidn" => {
                executable = Some(PathBuf::from(args.next().ok_or("missing OIDN executable")?))
            }
            "--limit" => limit = args.next().ok_or("missing limit")?.parse()?,
            _ => return Err(format!("unknown option {flag}").into()),
        }
    }
    let executable = executable.ok_or("--oidn required")?;
    let out = out.ok_or("--out required")?;
    if data.is_empty() || limit == 0 {
        return Err("nonempty --eval-data and positive limit required".into());
    }
    std::fs::create_dir(&out)?;
    let scratch = out.join("scratch");
    std::fs::create_dir(&scratch)?;
    let mut rows = std::fs::File::create(out.join("frames.csv"))?;
    writeln!(rows, "sequence,frame,psnr,ssim,gradient_mse")?;
    let mut total = [0.0; 3];
    let mut count = 0;
    let mut sequence = 0;
    let mut expected_layout = None;
    for path in &data {
        let (mut reader, _) = open_capture(path)?;
        let layout = *reader.layout();
        let shape = (
            layout.lr_width,
            layout.lr_height,
            layout.scale,
            reader.sequence_length(),
        );
        if expected_layout.is_some_and(|expected| expected != shape) {
            return Err("capture sequence/extent mismatch".into());
        }
        expected_layout = Some(shape);
        let low = [layout.lr_width as usize, layout.lr_height as usize];
        let high = [layout.hr_width(), layout.hr_height()];
        for i in 0..reader.len().min(limit) {
            let sample = reader.sample(i)?;
            let (inputs, jitter) = observed(&sample, layout)?;
            for (name, rgb) in ["color", "albedo", "normal"].into_iter().zip(&inputs) {
                write_pfm(
                    std::io::BufWriter::new(std::fs::File::create(
                        scratch.join(format!("{name}.pfm")),
                    )?),
                    rgb,
                    low,
                )?;
            }
            let run = Command::new(&executable)
                .args([
                    "--device",
                    "cpu",
                    "--filter",
                    "RT",
                    "--quality",
                    "high",
                    "--threads",
                    "4",
                    "--affinity",
                    "0",
                    "--clean_aux",
                ])
                .arg("--hdr")
                .arg(scratch.join("color.pfm"))
                .arg("--alb")
                .arg(scratch.join("albedo.pfm"))
                .arg("--nrm")
                .arg(scratch.join("normal.pfm"))
                .arg("--output")
                .arg(scratch.join("output.pfm"))
                .output()?;
            if !run.status.success() {
                return Err(format!(
                    "OIDN failed: {} {}",
                    String::from_utf8_lossy(&run.stdout),
                    String::from_utf8_lossy(&run.stderr)
                )
                .into());
            }
            if count == 0 {
                std::fs::write(out.join("oidn-first-frame.log"), &run.stdout)?;
            }
            let native = read_pfm(std::fs::File::open(scratch.join("output.pfm"))?, low)?;
            let image = resample(&native, low, layout.scale as usize, jitter);
            let mut reference = vec![0.0; layout.hr_texels() * 3];
            for c in 0..3 {
                for (j, value) in sample
                    .hr_channel(&layout, Plane::Color, c)
                    .ok_or("missing reference")?
                    .iter()
                    .enumerate()
                {
                    reference[3 * j + c] = value.to_f32();
                }
            }
            if reference.iter().any(|v| !v.is_finite() || *v < 0.0)
                || reference.iter().all(|v| *v <= 1e-6)
            {
                return Err("invalid or entirely black reference".into());
            }
            let scores = [
                -10.0
                    * f64::from(metrics::error(&image, &reference))
                        .max(1e-20)
                        .log10(),
                f64::from(metrics::ssim(
                    &image,
                    &reference,
                    high[0] as usize,
                    high[1] as usize,
                )),
                metrics::gradient_error(&image, &reference, high[0] as usize, high[1] as usize),
            ];
            let seq = sequence + i / reader.sequence_length();
            let frame = i % reader.sequence_length();
            writeln!(
                rows,
                "{seq},{frame},{:.6},{:.9},{:.12}",
                scores[0], scores[1], scores[2]
            )?;
            for c in 0..3 {
                total[c] += scores[c];
            }
            for (name, rgb) in [("learned", &image), ("reference", &reference)] {
                let prefix = format!("{seq:03}-{frame:03}-{name}");
                save_png(&out.join(format!("{prefix}.png")), rgb, high)?;
                if linear {
                    save_linear(&out.join(format!("{prefix}.rgbf32")), rgb)?;
                }
            }
            count += 1;
            if count % 16 == 0 {
                println!(
                    "OIDN: {count} frames, mean PSNR {:.2} dB",
                    total[0] / count as f64
                );
            }
        }
        sequence += reader.len() / reader.sequence_length();
    }
    let report = serde_json::json!({
        "role":"external spatial reference", "algorithm":"OIDN RT HDR/high, CPU/4 threads, default automatic inputScale",
        "inputs":"unaltered native LR beauty; clean primary albedo=clamp(diffuse_albedo+specular_F0,0,1); normalized world normals",
        "resampling":"bilinear output upsampling after denoising, with input jitter removed",
        "limitations":"no temporal history or HR guides; not a temporally stable or matched super-resolution algorithm",
        "output_name":"learned is the common scorer's prediction filename, not our learned model",
        "metric_space":"x/(1+x), before sRGB and quantization", "speed_claim":false,
        "frames":count, "psnr":total[0]/count as f64, "ssim":total[1]/count as f64, "gradient_mse":total[2]/count as f64,
        "datasets":data, "oidn_executable":executable,
    });
    std::fs::write(
        out.join("quality.json"),
        serde_json::to_vec_pretty(&report)?,
    )?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn observations_do_not_use_reference_or_hr_guides() {
        let layout = Layout {
            scale: 2,
            lr_width: 1,
            lr_height: 1,
            lr_source: ommatidia::dataset::InputSource::PathTrace,
            lr_planes: [
                Plane::Color,
                Plane::DiffuseAlbedo,
                Plane::SpecularF0,
                Plane::Normal,
                Plane::Jitter,
            ]
            .into_iter()
            .collect(),
            hr_planes: [Plane::Color, Plane::Normal].into_iter().collect(),
        };
        let mut sample = Sample {
            lr: vec![Default::default(); layout.lr_len()],
            hr: vec![Default::default(); layout.hr_len()],
        };
        let original = observed(&sample, layout).unwrap();
        sample.hr.clear();
        assert_eq!(original, observed(&sample, layout).unwrap());
    }

    #[test]
    fn pfm_preserves_orientation_and_hdr() {
        let image = vec![1.0, 2.0, 3.0, 1000.0, 20.0, 30.0];
        let mut bytes = Vec::new();
        write_pfm(&mut bytes, &image, [1, 2]).unwrap();
        assert!(bytes.starts_with(b"PF\n1 2\n-1.0\n"));
        assert_eq!(&bytes[12..16], &1000.0f32.to_le_bytes());
        assert_eq!(read_pfm(bytes.as_slice(), [1, 2]).unwrap(), image);
        assert!(read_pfm(bytes.as_slice(), [2, 1]).is_err());
    }
    #[test]
    fn resampling_removes_jitter_and_preserves_constants() {
        assert_eq!(
            resample(&[2.0; 12], [2, 2], 2, [0.25, -0.25]),
            vec![2.0; 48]
        );
        let ramp = [
            0.0, 0.0, 0.0, 4.0, 4.0, 4.0, 8.0, 8.0, 8.0, 12.0, 12.0, 12.0,
        ];
        let regular = resample(&ramp, [2, 2], 2, [0.0; 2]);
        assert_eq!(regular[3], 1.0);
        let jittered = resample(&ramp, [2, 2], 2, [0.25, -0.25]);
        assert_eq!(jittered[3], 0.0);
        assert_eq!(jittered[9], 4.0);
    }
}
