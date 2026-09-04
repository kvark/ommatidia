//! Measure the finite-sample noise floor from two independent references.

use std::path::PathBuf;

use ommatidia::dataset::{Layout, Plane, Reader, Sample};

struct Args {
    a: PathBuf,
    b: PathBuf,
    limit: usize,
}

fn usage() -> &'static str {
    "measure canonical reference noise\n\n\
usage: reference-noise --a PATH --b PATH [--limit N]\n\n\
The files must describe the same scenes in the same order. A shorter file may\n\
check a shared prefix. Generate it with the same seed and an offset at least as\n\
large as --canonical-frames, so its paths do not\n\
overlap the first reference."
}

fn parse_args() -> Result<Args, String> {
    let mut a = None;
    let mut b = None;
    let mut limit = usize::MAX;
    let mut args = std::env::args().skip(1);
    while let Some(flag) = args.next() {
        let mut value = || args.next().ok_or_else(|| format!("{flag} needs a value"));
        match flag.as_str() {
            "--a" => a = Some(PathBuf::from(value()?)),
            "--b" => b = Some(PathBuf::from(value()?)),
            "--limit" => limit = value()?.parse().map_err(|e| format!("--limit: {e}"))?,
            "-h" | "--help" => {
                println!("{}", usage());
                std::process::exit(0);
            }
            other => return Err(format!("unknown option {other:?}\n\n{}", usage())),
        }
    }
    Ok(Args {
        a: a.ok_or_else(|| format!("--a is required\n\n{}", usage()))?,
        b: b.ok_or_else(|| format!("--b is required\n\n{}", usage()))?,
        limit,
    })
}

fn interleaved_color(sample: &Sample, layout: &Layout) -> Vec<f32> {
    let texels = layout.hr_texels();
    let mut out = vec![0.0; texels * 3];
    for channel in 0..3 {
        let source = sample
            .hr_channel(layout, Plane::Color, channel)
            .expect("reference has no high-resolution color");
        for (index, value) in source.iter().enumerate() {
            out[index * 3 + channel] = value.to_f32();
        }
    }
    out
}

fn psnr(mse: f64) -> f64 {
    if mse == 0.0 {
        f64::INFINITY
    } else {
        -10.0 * mse.log10()
    }
}

fn main() {
    let args = parse_args().unwrap_or_else(|message| {
        eprintln!("{message}");
        std::process::exit(2);
    });
    let mut a = Reader::open(&args.a)
        .unwrap_or_else(|error| panic!("cannot open {}: {error}", args.a.display()));
    let mut b = Reader::open(&args.b)
        .unwrap_or_else(|error| panic!("cannot open {}: {error}", args.b.display()));
    assert_eq!(a.layout(), b.layout(), "reference layouts differ");
    assert_eq!(
        a.sequence_length(),
        b.sequence_length(),
        "reference sequence lengths differ"
    );
    let layout = *a.layout();
    let count = a.len().min(b.len()).min(args.limit);
    assert!(count != 0, "no records to compare");

    let mut mse = 0.0f64;
    let mut relative = 0.0f64;
    let mut low_frequency = 0.0f64;
    let mut ssim = 0.0f64;
    let mut detail_a = 0.0f64;
    let mut detail_b = 0.0f64;
    let mut energy_a = 0.0f64;
    let mut energy_b = 0.0f64;
    for index in 0..count {
        let sample_a = a.sample(index).expect("cannot read reference A");
        let sample_b = b.sample(index).expect("cannot read reference B");
        assert_eq!(
            sample_a.lr, sample_b.lr,
            "record {index} has different input/G-buffer data; the scenes are not matched"
        );
        let image_a = interleaved_color(&sample_a, &layout);
        let image_b = interleaved_color(&sample_b, &layout);
        mse += f64::from(ommatidia::metrics::error(&image_a, &image_b));
        relative += ommatidia::metrics::relative_error(&image_a, &image_b);
        low_frequency += ommatidia::metrics::low_frequency_error(
            &image_a,
            &image_b,
            layout.hr_width() as usize,
            layout.hr_height() as usize,
            16,
        );
        ssim += f64::from(ommatidia::metrics::ssim(
            &image_a,
            &image_b,
            layout.hr_width() as usize,
            layout.hr_height() as usize,
        ));
        detail_a += ommatidia::metrics::detail(
            &image_a,
            layout.hr_width() as usize,
            layout.hr_height() as usize,
        );
        detail_b += ommatidia::metrics::detail(
            &image_b,
            layout.hr_width() as usize,
            layout.hr_height() as usize,
        );
        energy_a += image_a.iter().map(|&value| f64::from(value)).sum::<f64>();
        energy_b += image_b.iter().map(|&value| f64::from(value)).sum::<f64>();
    }
    let count = count as f64;
    mse /= count;
    relative /= count;
    low_frequency /= count;
    ssim /= count;
    detail_a /= count;
    detail_b /= count;

    // Independent equal-variance estimates differ with variance 2 sigma^2.
    // Model-versus-one-reference scores therefore encounter half the measured
    // pairwise MSE from the reference itself (3.01 dB above pair PSNR).
    println!("independent references over {count:.0} matched records:");
    println!("  pairwise PSNR       {:.2} dB", psnr(mse));
    println!(
        "  one-reference floor {:.2} dB (half the pair variance)",
        psnr(0.5 * mse)
    );
    println!("  SSIM                {ssim:.6}");
    println!("  relative MSE        {relative:.8}");
    println!("  low-frequency PSNR {:.2} dB", psnr(low_frequency));
    println!(
        "  mean energy B/A     {:.6} ({:+.3}%)",
        energy_b / energy_a,
        100.0 * (energy_b / energy_a - 1.0)
    );
    println!("  detail B/A          {:.6}", detail_b / detail_a);
}
