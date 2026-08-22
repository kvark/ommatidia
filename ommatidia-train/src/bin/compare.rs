//! Reconstruct named, deterministic dataset samples with every comparison arm.
//!
//! OIDN is deliberately an external executable: it is a benchmark baseline,
//! not a runtime dependency. The official `oidnDenoise` binary accepts PFM,
//! which keeps the exchange linear HDR without adding an image-format crate.

use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::process::Command;

use ommatidia::batch::{self, Crop, ExtraTaps};
use ommatidia::dataset::{Layout, Plane, Reader, Sample};
use ommatidia::model::{Objective, Prediction};

#[derive(Debug)]
struct Case {
    name: String,
    index: usize,
}

struct Args {
    data: PathBuf,
    restir_svgf_data: Option<PathBuf>,
    checkpoint: PathBuf,
    oidn: Option<PathBuf>,
    oidn_device: String,
    device_id: Option<u32>,
    out: PathBuf,
    cases: Vec<Case>,
}

impl Default for Args {
    fn default() -> Self {
        Self {
            data: PathBuf::from("data/rich-4spp-validation-128.omd"),
            restir_svgf_data: None,
            checkpoint: PathBuf::from("runs/rich-kernel-b16-demod025"),
            oidn: None,
            oidn_device: "default".into(),
            device_id: None,
            out: PathBuf::from("runs/comparison-suite"),
            cases: Vec::new(),
        }
    }
}

fn usage() -> &'static str {
    "compare Ommatidium, OIDN, and matched ReSTIR+SVGF\n\n\
usage: compare [options]\n\n\
  --data PATH                independent sparse-path dataset\n\
  --restir-svgf-data PATH    matched Blade ReSTIR+SVGF dataset\n\
  --checkpoint STEM          Ommatidium checkpoint stem\n\
  --oidn PATH                official oidnDenoise executable\n\
  --oidn-device ID           oidnDenoise physical device ID [default]\n\
  --device-id ID             Vulkan PCI device ID for this standalone tool\n\
  --out DIR                  comparison artifact directory\n\
  --case NAME:INDEX          named sample; repeat for a suite\n\
  -h, --help                 show this text"
}

fn parse_args() -> Result<Args, String> {
    let mut out = Args::default();
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        let mut value = || args.next().ok_or_else(|| format!("{arg} needs a value"));
        match arg.as_str() {
            "--data" => out.data = value()?.into(),
            "--restir-svgf-data" => out.restir_svgf_data = Some(value()?.into()),
            "--checkpoint" => out.checkpoint = value()?.into(),
            "--oidn" => out.oidn = Some(value()?.into()),
            "--oidn-device" => out.oidn_device = value()?,
            "--device-id" => out.device_id = Some(ommatidia::gpu::parse_device_id(&value()?)?),
            "--out" => out.out = value()?.into(),
            "--case" => {
                let value = value()?;
                let (name, index) = value
                    .split_once(':')
                    .ok_or_else(|| format!("case {value:?} must be NAME:INDEX"))?;
                if name.is_empty()
                    || !name
                        .chars()
                        .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-')
                {
                    return Err(format!(
                        "case name {name:?} must use lowercase letters, digits, and hyphens"
                    ));
                }
                out.cases.push(Case {
                    name: name.into(),
                    index: index
                        .parse()
                        .map_err(|e| format!("invalid case index {index:?}: {e}"))?,
                });
            }
            "-h" | "--help" => {
                println!("{}", usage());
                std::process::exit(0);
            }
            other => return Err(format!("unknown option {other:?}\n\n{}", usage())),
        }
    }
    if out.cases.is_empty() {
        return Err(format!("at least one --case is required\n\n{}", usage()));
    }
    Ok(out)
}

fn interleaved(sample: &Sample, layout: &Layout, plane: Plane, high: bool) -> Vec<f32> {
    let (width, height) = if high {
        (layout.hr_width(), layout.hr_height())
    } else {
        (layout.lr_width, layout.lr_height)
    };
    let texels = width as usize * height as usize;
    let mut out = vec![0.0; texels * 3];
    for channel in 0..3 {
        let source = if high {
            sample.hr_channel(layout, plane, channel)
        } else {
            sample.lr_channel(layout, plane, channel)
        }
        .unwrap_or_else(|| panic!("dataset has no {plane:?} {width}x{height} channel {channel}"));
        for (index, value) in source.iter().enumerate() {
            out[index * 3 + channel] = value.to_f32();
        }
    }
    out
}

fn bilinear(input: &[f32], width: usize, height: usize, scale: usize) -> Vec<f32> {
    let mut output = vec![0.0; width * scale * height * scale * 3];
    for oy in 0..height * scale {
        let fy = (oy as f32 + 0.5) / scale as f32 - 0.5;
        let y0_raw = fy.floor() as isize;
        let ty = fy - y0_raw as f32;
        let y0 = y0_raw.clamp(0, height as isize - 1) as usize;
        let y1 = (y0_raw + 1).clamp(0, height as isize - 1) as usize;
        for ox in 0..width * scale {
            let fx = (ox as f32 + 0.5) / scale as f32 - 0.5;
            let x0_raw = fx.floor() as isize;
            let tx = fx - x0_raw as f32;
            let x0 = x0_raw.clamp(0, width as isize - 1) as usize;
            let x1 = (x0_raw + 1).clamp(0, width as isize - 1) as usize;
            for c in 0..3 {
                let at = |x, y| input[(y * width + x) * 3 + c];
                let top = at(x0, y0) + tx * (at(x1, y0) - at(x0, y0));
                let bottom = at(x0, y1) + tx * (at(x1, y1) - at(x0, y1));
                output[(oy * width * scale + ox) * 3 + c] = top + ty * (bottom - top);
            }
        }
    }
    output
}

fn write_png(path: &Path, rgb: &[f32], width: usize, height: usize) -> Result<(), String> {
    let mut bytes = Vec::with_capacity(width * height * 4);
    for texel in rgb.chunks_exact(3) {
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
    let file = File::create(path).map_err(|e| e.to_string())?;
    let mut encoder = png::Encoder::new(file, width as u32, height as u32);
    encoder.set_color(png::ColorType::Rgba);
    encoder.set_depth(png::BitDepth::Eight);
    encoder
        .write_header()
        .and_then(|mut writer| writer.write_image_data(&bytes))
        .map_err(|e| e.to_string())
}

fn write_pfm(path: &Path, rgb: &[f32], width: usize, height: usize) -> Result<(), String> {
    if rgb.len() != width * height * 3 {
        return Err("PFM image extent does not match its data".into());
    }
    let mut file = BufWriter::new(File::create(path).map_err(|e| e.to_string())?);
    write!(file, "PF\n{width} {height}\n-1.0\n").map_err(|e| e.to_string())?;
    for row in (0..height).rev() {
        for &value in &rgb[row * width * 3..(row + 1) * width * 3] {
            file.write_all(&value.to_le_bytes())
                .map_err(|e| e.to_string())?;
        }
    }
    Ok(())
}

fn read_pfm(path: &Path) -> Result<(Vec<f32>, usize, usize), String> {
    let mut file = BufReader::new(File::open(path).map_err(|e| e.to_string())?);
    let mut line = String::new();
    file.read_line(&mut line).map_err(|e| e.to_string())?;
    if line.trim() != "PF" {
        return Err(format!("{} is not an RGB PFM", path.display()));
    }
    line.clear();
    file.read_line(&mut line).map_err(|e| e.to_string())?;
    let mut extent = line.split_whitespace();
    let width: usize = extent
        .next()
        .ok_or("PFM width is missing")?
        .parse()
        .map_err(|e| format!("invalid PFM width: {e}"))?;
    let height: usize = extent
        .next()
        .ok_or("PFM height is missing")?
        .parse()
        .map_err(|e| format!("invalid PFM height: {e}"))?;
    line.clear();
    file.read_line(&mut line).map_err(|e| e.to_string())?;
    let scale: f32 = line
        .trim()
        .parse()
        .map_err(|e| format!("invalid PFM scale: {e}"))?;
    if scale >= 0.0 {
        return Err("big-endian PFM output is not supported".into());
    }
    let mut bytes = vec![0; width * height * 3 * 4];
    file.read_exact(&mut bytes).map_err(|e| e.to_string())?;
    let mut rgb = vec![0.0; width * height * 3];
    for row in 0..height {
        let source_row = height - 1 - row;
        for x in 0..width * 3 {
            let offset = (source_row * width * 3 + x) * 4;
            rgb[row * width * 3 + x] = f32::from_le_bytes(
                bytes[offset..offset + 4]
                    .try_into()
                    .expect("four-byte chunk"),
            );
        }
    }
    Ok((rgb, width, height))
}

struct OidnInput<'a> {
    color: &'a [f32],
    albedo: &'a [f32],
    normal: &'a [f32],
    width: usize,
    height: usize,
}

fn oidn(
    executable: &Path,
    device: &str,
    quality: &str,
    dir: &Path,
    input: OidnInput<'_>,
) -> Result<Vec<f32>, String> {
    let OidnInput {
        color,
        albedo,
        normal,
        width,
        height,
    } = input;
    std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
    let color_path = dir.join("color.pfm");
    let albedo_path = dir.join("albedo.pfm");
    let normal_path = dir.join("normal.pfm");
    let output_path = dir.join("output.pfm");
    write_pfm(&color_path, color, width, height)?;
    write_pfm(&albedo_path, albedo, width, height)?;
    write_pfm(&normal_path, normal, width, height)?;
    let output = Command::new(executable)
        .args(["--device", device, "--hdr"])
        .arg(&color_path)
        .arg("--alb")
        .arg(&albedo_path)
        .arg("--nrm")
        .arg(&normal_path)
        .args(["--clean_aux", "--quality", quality, "--output"])
        .arg(&output_path)
        .output()
        .map_err(|e| format!("cannot execute {}: {e}", executable.display()))?;
    if !output.status.success() {
        return Err(format!(
            "{} failed:\n{}{}",
            executable.display(),
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr),
        ));
    }
    let result = read_pfm(&output_path);
    // PFM is an exact interchange format, not a suite artifact. Keep only the
    // display PNG and metrics that a reviewer needs.
    let _ = std::fs::remove_dir_all(dir);
    let (image, out_width, out_height) = result?;
    if [out_width, out_height] != [width, height] {
        return Err(format!(
            "OIDN returned {out_width}x{out_height}, expected {width}x{height}"
        ));
    }
    Ok(image)
}

fn mean_luminance(image: &[f32]) -> f64 {
    image
        .chunks_exact(3)
        .map(|rgb| 0.2126 * rgb[0] as f64 + 0.7152 * rgb[1] as f64 + 0.0722 * rgb[2] as f64)
        .sum::<f64>()
        / (image.len() / 3).max(1) as f64
}

fn metric_row(case: &str, method: &str, image: &[f32], reference: &[f32], width: usize) -> String {
    let mse = ommatidia::metrics::error(image, reference) as f64;
    let psnr = -10.0 * mse.log10();
    let ssim = ommatidia::metrics::ssim(image, reference, width, width);
    let relative = ommatidia::metrics::relative_error(image, reference);
    let detail = ommatidia::metrics::detail(image, width, width)
        / ommatidia::metrics::detail(reference, width, width);
    let energy = mean_luminance(image) / mean_luminance(reference);
    let low_frequency = ommatidia::metrics::low_frequency_error(image, reference, width, width, 16);
    let low_frequency_psnr = -10.0 * low_frequency.log10();
    format!(
        "{case},{method},{mse:.8},{psnr:.3},{ssim:.6},{relative:.8},{detail:.6},\
         {energy:.6},{low_frequency:.8},{low_frequency_psnr:.3}\n"
    )
}

fn reconstruct(
    session: &mut meganeura::Session,
    config: &ommatidia::model::ModelConfig,
    sample: &Sample,
    layout: &Layout,
) -> Vec<f32> {
    let crop = Crop {
        x: 0,
        y: 0,
        tile: config.tile,
    };
    let mut conditioning = vec![0.0; config.cond_len()];
    batch::write_conditioning(
        sample,
        layout,
        config.cond_planes,
        crop,
        0,
        &mut conditioning,
    );
    session.set_input("cond", &conditioning);
    session.step();
    session.wait();
    let weights = session.read_output(config.target_len());
    batch::assemble_kernel(sample, layout, crop, &weights, config, ExtraTaps::default())
}

fn run(args: Args) -> Result<(), String> {
    let mut reader = Reader::open(&args.data).map_err(|e| e.to_string())?;
    let layout = *reader.layout();
    if layout.lr_width != layout.lr_height {
        return Err("comparison runner currently requires square dataset frames".into());
    }
    for plane in [Plane::Color, Plane::Normal, Plane::DiffuseAlbedo] {
        if !layout.lr_planes.contains(plane) || !layout.hr_planes.contains(plane) {
            return Err(format!(
                "comparison needs low- and high-resolution {plane:?}"
            ));
        }
    }

    let mut restir_reader = args
        .restir_svgf_data
        .as_ref()
        .map(Reader::open)
        .transpose()
        .map_err(|e| e.to_string())?;
    if let Some(restir) = &restir_reader {
        let control = restir.layout();
        if control.scale != layout.scale
            || control.lr_width != layout.lr_width
            || control.lr_height != layout.lr_height
            || control.lr_planes != layout.lr_planes
            || control.hr_planes != layout.hr_planes
        {
            return Err("ReSTIR+SVGF layout does not match the path-traced dataset".into());
        }
    }

    let (mut config, paths) = ommatidia::checkpoint::load_config(&args.checkpoint)
        .map_err(|e| format!("cannot load {}: {e}", args.checkpoint.display()))?;
    if config.objective != Objective::Direct || config.prediction != Prediction::SubpixelKernel {
        return Err("comparison currently requires a direct sub-pixel-kernel checkpoint".into());
    }
    if config.temporal.is_some() {
        return Err("comparison scenes are static; use a spatial checkpoint".into());
    }
    config.tile = layout.lr_width;
    config.batch = 1;
    config.validate()?;
    let model = ommatidia::model::build(&config, false)?;
    let context = ommatidia::gpu::create_context(args.device_id, false);
    println!(
        "Ommatidium on {}: {} parameters, {:.1} GFLOP at suite extent",
        context.device_information().device_name,
        model.params.iter().map(|param| param.len).sum::<usize>(),
        config.flops(layout.hr_texels()),
    );
    let mut session = ommatidia::gpu::inference_session(&model.graph, context);
    session
        .load_checkpoint(&paths.weights)
        .map_err(|e| format!("cannot load {}: {e}", paths.weights.display()))?;

    std::fs::create_dir_all(&args.out).map_err(|e| e.to_string())?;
    let metadata = format!(
        "data={}\nrestir_svgf_data={}\ncheckpoint={}\noidn={}\noidn_device={}\n",
        args.data.display(),
        args.restir_svgf_data
            .as_deref()
            .map_or_else(|| "none".into(), |path| path.display().to_string()),
        args.checkpoint.display(),
        args.oidn
            .as_deref()
            .and_then(Path::file_name)
            .map_or_else(|| "none".into(), |name| name.to_string_lossy().into_owned()),
        args.oidn_device,
    );
    std::fs::write(args.out.join("metadata.txt"), metadata).map_err(|e| e.to_string())?;
    let mut csv = String::from(
        "case,method,mse,psnr_db,ssim,relative_mse,detail_ratio,energy_ratio,\
         low_frequency_mse,low_frequency_psnr_db\n",
    );
    let lr_width = layout.lr_width as usize;
    let lr_height = layout.lr_height as usize;
    let scale = layout.scale as usize;
    let hr_width = layout.hr_width() as usize;
    let hr_height = layout.hr_height() as usize;

    for case in &args.cases {
        let sample = reader.sample(case.index).map_err(|e| e.to_string())?;
        let low = interleaved(&sample, &layout, Plane::Color, false);
        let bilinear_image = bilinear(&low, lr_width, lr_height, scale);
        let predicted = reconstruct(&mut session, &config, &sample, &layout);
        let reference = interleaved(&sample, &layout, Plane::Color, true);
        let case_dir = args.out.join(&case.name);
        std::fs::create_dir_all(&case_dir).map_err(|e| e.to_string())?;

        let mut methods: Vec<(&str, Vec<f32>)> = vec![
            ("bilinear", bilinear_image.clone()),
            ("ommatidium", predicted),
        ];
        if let Some(restir) = &mut restir_reader {
            let control = restir.sample(case.index).map_err(|e| e.to_string())?;
            if control.hr != sample.hr {
                return Err(format!(
                    "case {} has a different canonical record in ReSTIR+SVGF data",
                    case.name
                ));
            }
            let color = interleaved(&control, &layout, Plane::Color, false);
            methods.push(("restir-svgf", bilinear(&color, lr_width, lr_height, scale)));
        }
        if let Some(executable) = &args.oidn {
            let low_albedo = interleaved(&sample, &layout, Plane::DiffuseAlbedo, false);
            let low_normal = interleaved(&sample, &layout, Plane::Normal, false);
            let oidn_low = oidn(
                executable,
                &args.oidn_device,
                "high",
                &case_dir.join("oidn-low-work"),
                OidnInput {
                    color: &low,
                    albedo: &low_albedo,
                    normal: &low_normal,
                    width: lr_width,
                    height: lr_height,
                },
            )?;
            methods.push((
                "oidn-input-high",
                bilinear(&oidn_low, lr_width, lr_height, scale),
            ));
            let oidn_low_fast = oidn(
                executable,
                &args.oidn_device,
                "fast",
                &case_dir.join("oidn-low-fast-work"),
                OidnInput {
                    color: &low,
                    albedo: &low_albedo,
                    normal: &low_normal,
                    width: lr_width,
                    height: lr_height,
                },
            )?;
            methods.push((
                "oidn-input-fast",
                bilinear(&oidn_low_fast, lr_width, lr_height, scale),
            ));

            let high_albedo = interleaved(&sample, &layout, Plane::DiffuseAlbedo, true);
            let high_normal = interleaved(&sample, &layout, Plane::Normal, true);
            let oidn_high = oidn(
                executable,
                &args.oidn_device,
                "high",
                &case_dir.join("oidn-high-work"),
                OidnInput {
                    color: &bilinear_image,
                    albedo: &high_albedo,
                    normal: &high_normal,
                    width: hr_width,
                    height: hr_height,
                },
            )?;
            methods.push(("oidn-output-high", oidn_high));
        }

        write_png(&case_dir.join("input.png"), &low, lr_width, lr_height)?;
        write_png(
            &case_dir.join("canonical.png"),
            &reference,
            hr_width,
            hr_height,
        )?;
        for (name, image) in methods {
            write_png(
                &case_dir.join(format!("{name}.png")),
                &image,
                hr_width,
                hr_height,
            )?;
            csv.push_str(&metric_row(&case.name, name, &image, &reference, hr_width));
        }
        println!("wrote {} (dataset sample {})", case.name, case.index);
    }
    std::fs::write(args.out.join("metrics.csv"), csv).map_err(|e| e.to_string())?;
    Ok(())
}

fn main() {
    env_logger::init();
    let result = parse_args().and_then(run);
    if let Err(error) = result {
        eprintln!("{error}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pfm_round_trip_preserves_orientation_and_hdr() {
        let path = std::env::temp_dir().join(format!(
            "ommatidia-pfm-{}-{}.pfm",
            std::process::id(),
            std::thread::current().name().unwrap_or("test")
        ));
        let image = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, // top row
            7.0, 8.0, 9.0, 10.0, 11.0, 1000.0, // bottom row
        ];
        write_pfm(&path, &image, 2, 2).unwrap();
        let (back, width, height) = read_pfm(&path).unwrap();
        assert_eq!([width, height], [2, 2]);
        assert_eq!(back, image);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn bilinear_aligns_texel_centres() {
        let input = vec![0.0, 0.0, 0.0, 4.0, 8.0, 12.0];
        let output = bilinear(&input, 2, 1, 2);
        let red: Vec<_> = output.chunks_exact(3).map(|rgb| rgb[0]).collect();
        assert_eq!(red, vec![0.0, 1.0, 3.0, 4.0, 0.0, 1.0, 3.0, 4.0]);
    }
}
