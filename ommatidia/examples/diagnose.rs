//! What the reported number does not say.
//!
//! Scores the deterministic reconstruction the product actually ships against
//! the canonical reference, in the project's own compressed space and in the
//! space the frame is displayed in, and splits the error by where it lives.
//!
//! Runs on the CPU only, so it can be used while the GPU is busy.

use std::path::PathBuf;

use ommatidia::batch::{self, Crop};
use ommatidia::dataset::{Layout, Plane, Reader, Sample};
use ommatidia::model::GuideConfig;
use ommatidia::transform;

/// Exactly the transform `eval::write_png` applies, so this scores the pixels
/// that end up on the screen.
fn display(x: f32) -> f32 {
    let mapped = transform::compress(x);
    if mapped <= 0.0031308 {
        12.92 * mapped
    } else {
        1.055 * mapped.powf(1.0 / 2.4) - 0.055
    }
    .clamp(0.0, 1.0)
}

fn luminance(rgb: &[f32]) -> f32 {
    0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
}

fn psnr(mse: f64) -> f64 {
    -10.0 * mse.log10()
}

/// SSIM as Wang et al. define it: an 11x11 Gaussian window at every pixel,
/// sigma 1.5. The project's own `metrics::ssim` instead averages independent
/// 8x8 blocks, which cannot see any structure that straddles a block edge.
fn gaussian_ssim(a: &[f32], b: &[f32], extent: usize) -> f64 {
    const C1: f64 = 0.0001;
    const C2: f64 = 0.0009;
    let kernel: Vec<f64> = (0..11)
        .map(|i| {
            let d = i as f64 - 5.0;
            (-d * d / (2.0 * 1.5 * 1.5)).exp()
        })
        .collect();
    let norm: f64 = kernel.iter().sum();
    let compressed_luma = |image: &[f32]| -> Vec<f64> {
        (0..extent * extent)
            .map(|i| {
                let rgb = &image[i * 3..];
                0.2126 * transform::compress(rgb[0]) as f64
                    + 0.7152 * transform::compress(rgb[1]) as f64
                    + 0.0722 * transform::compress(rgb[2]) as f64
            })
            .collect()
    };
    let luma = compressed_luma(a);
    let luma_b = compressed_luma(b);

    let mut total = 0.0;
    let mut count = 0usize;
    for cy in 5..extent - 5 {
        for cx in 5..extent - 5 {
            let (mut ma, mut mb, mut w) = (0.0, 0.0, 0.0);
            for ky in 0..11 {
                for kx in 0..11 {
                    let weight = kernel[ky] * kernel[kx] / (norm * norm);
                    let i = (cy + ky - 5) * extent + cx + kx - 5;
                    ma += weight * luma[i];
                    mb += weight * luma_b[i];
                    w += weight;
                }
            }
            ma /= w;
            mb /= w;
            let (mut va, mut vb, mut cov) = (0.0, 0.0, 0.0);
            for ky in 0..11 {
                for kx in 0..11 {
                    let weight = kernel[ky] * kernel[kx] / (norm * norm) / w;
                    let i = (cy + ky - 5) * extent + cx + kx - 5;
                    let da = luma[i] - ma;
                    let db = luma_b[i] - mb;
                    va += weight * da * da;
                    vb += weight * db * db;
                    cov += weight * da * db;
                }
            }
            total += ((2.0 * ma * mb + C1) * (2.0 * cov + C2))
                / ((ma * ma + mb * mb + C1) * (va + vb + C2));
            count += 1;
        }
    }
    total / count as f64
}

#[derive(Default, Clone, Copy)]
struct Score {
    compressed: f64,
    displayed: f64,
    relative: f64,
    values: usize,
}

impl Score {
    fn accumulate(&mut self, a: &[f32], b: &[f32]) {
        for (&x, &y) in a.iter().zip(b.iter()) {
            let dc = transform::compress(x) - transform::compress(y);
            self.compressed += (dc * dc) as f64;
            let dd = display(x) - display(y);
            self.displayed += (dd * dd) as f64;
            // Rousselle's relative MSE: the rendering metric that does not let
            // a bright region decide the score for the whole frame.
            let d = x - y;
            self.relative += (d * d / (y * y + 0.01)) as f64;
            self.values += 1;
        }
    }

    fn report(&self, name: &str) {
        let n = self.values as f64;
        println!(
            "  {name:<26} PSNR {:6.2} dB   display PSNR {:6.2} dB   relMSE {:.5}",
            psnr(self.compressed / n),
            psnr(self.displayed / n),
            self.relative / n,
        );
    }
}

/// Box-downsample the canonical reference to input resolution: the colour a
/// perfect denoiser would hand the upsampler.
fn oracle_low_resolution(sample: &Sample, layout: &Layout) -> Vec<f32> {
    let scale = layout.scale as usize;
    let width = layout.lr_width as usize;
    let height = layout.lr_height as usize;
    let hr_width = layout.hr_width() as usize;
    let base = layout
        .hr_planes
        .channel_offset(Plane::Color)
        .expect("no high resolution colour");
    let texels = layout.hr_texels();
    let mut out = vec![0.0; width * height * 3];
    for c in 0..3 {
        let source = &sample.hr[(base + c) * texels..(base + c + 1) * texels];
        for y in 0..height {
            for x in 0..width {
                let mut sum = 0.0;
                for dy in 0..scale {
                    for dx in 0..scale {
                        sum += source[(y * scale + dy) * hr_width + x * scale + dx].to_f32();
                    }
                }
                out[(y * width + x) * 3 + c] = sum / (scale * scale) as f32;
            }
        }
    }
    out
}

/// Separable box blur of an interleaved image, in compressed space, radius in
/// output pixels. Used both to ask whether the metric rewards blur and to split
/// the error into a smooth and a noisy part.
fn blur(image: &[f32], extent: usize, radius: usize) -> Vec<f32> {
    let mut mid = vec![0.0f32; image.len()];
    let mut out = vec![0.0f32; image.len()];
    for y in 0..extent {
        for x in 0..extent {
            for c in 0..3 {
                let mut sum = 0.0;
                let mut count = 0.0;
                for dx in -(radius as i32)..=radius as i32 {
                    let sx = (x as i32 + dx).clamp(0, extent as i32 - 1) as usize;
                    sum += transform::compress(image[(y * extent + sx) * 3 + c]);
                    count += 1.0;
                }
                mid[(y * extent + x) * 3 + c] = sum / count;
            }
        }
    }
    for y in 0..extent {
        for x in 0..extent {
            for c in 0..3 {
                let mut sum = 0.0;
                let mut count = 0.0;
                for dy in -(radius as i32)..=radius as i32 {
                    let sy = (y as i32 + dy).clamp(0, extent as i32 - 1) as usize;
                    sum += mid[(sy * extent + x) * 3 + c];
                    count += 1.0;
                }
                out[(y * extent + x) * 3 + c] = transform::decompress(sum / count);
            }
        }
    }
    out
}

/// Interleaved albedo for one crop, at output resolution.
fn crop_hr_albedo(sample: &Sample, layout: &Layout, crop: Crop) -> Vec<f32> {
    let scale = layout.scale as usize;
    let extent = crop.tile as usize * scale;
    let stride = layout.hr_width() as usize;
    let mut out = vec![0.0; extent * extent * 3];
    for c in 0..3 {
        let plane = sample
            .hr_channel(layout, Plane::DiffuseAlbedo, c)
            .expect("no high resolution albedo");
        for y in 0..extent {
            for x in 0..extent {
                let source = (crop.y as usize * scale + y) * stride + crop.x as usize * scale + x;
                out[(y * extent + x) * 3 + c] = plane[source].to_f32();
            }
        }
    }
    out
}

/// Whole-frame albedo at input resolution, interleaved.
fn lr_albedo(sample: &Sample, layout: &Layout) -> Vec<f32> {
    let texels = layout.lr_texels();
    let mut out = vec![0.0; texels * 3];
    for c in 0..3 {
        let plane = sample
            .lr_channel(layout, Plane::DiffuseAlbedo, c)
            .expect("no low resolution albedo");
        for i in 0..texels {
            out[i * 3 + c] = plane[i].to_f32();
        }
    }
    out
}

/// Divide radiance by albedo before reconstruction and multiply the exact
/// output-resolution albedo back afterwards. The offset is the same on both
/// sides, so a surface whose albedo does not change across the upsample comes
/// back bit-for-bit; only boundaries move.
const DEMODULATION_OFFSET: f32 = 0.05;

fn demodulate(color: &[f32], albedo: &[f32]) -> Vec<f32> {
    color
        .iter()
        .zip(albedo.iter())
        .map(|(&c, &a)| c / (a + DEMODULATION_OFFSET))
        .collect()
}

fn remodulate(color: &[f32], albedo: &[f32]) -> Vec<f32> {
    color
        .iter()
        .zip(albedo.iter())
        .map(|(&c, &a)| c * (a + DEMODULATION_OFFSET))
        .collect()
}

/// Mean gradient magnitude of displayed luminance: how much detail an image
/// carries, in the space it is looked at.
fn detail(image: &[f32], extent: usize) -> f64 {
    let at = |x: usize, y: usize| display(luminance(&image[(y * extent + x) * 3..])) as f64;
    let mut sum = 0.0;
    for y in 0..extent - 1 {
        for x in 0..extent - 1 {
            let c = at(x, y);
            sum += ((at(x + 1, y) - c).powi(2) + (at(x, y + 1) - c).powi(2)).sqrt();
        }
    }
    sum / ((extent - 1) * (extent - 1)) as f64
}

/// Mark output pixels next to a depth or normal discontinuity in the
/// high-resolution primary surface.
fn silhouette_mask(sample: &Sample, layout: &Layout, crop: Crop) -> Vec<bool> {
    let scale = layout.scale as usize;
    let extent = crop.tile as usize * scale;
    let stride = layout.hr_width() as usize;
    let origin_x = crop.x as usize * scale;
    let origin_y = crop.y as usize * scale;
    let depth = sample.hr_channel(layout, Plane::Depth, 0);
    let normal: Vec<_> = (0..3)
        .map(|c| sample.hr_channel(layout, Plane::Normal, c))
        .collect();
    let mut mask = vec![false; extent * extent];
    let (Some(depth), true) = (depth, normal.iter().all(Option::is_some)) else {
        return mask;
    };
    let normal: Vec<&[half::f16]> = normal.into_iter().map(Option::unwrap).collect();
    for y in 0..extent {
        for x in 0..extent {
            let here = (origin_y + y) * stride + origin_x + x;
            let d0 = depth[here].to_f32();
            let n0: Vec<f32> = normal.iter().map(|p| p[here].to_f32()).collect();
            let mut edge = false;
            for (dx, dy) in [(1i32, 0i32), (0, 1), (-1, 0), (0, -1)] {
                let nx = (origin_x + x) as i32 + dx;
                let ny = (origin_y + y) as i32 + dy;
                if nx < 0 || ny < 0 || nx >= stride as i32 || ny >= layout.hr_height() as i32 {
                    continue;
                }
                let there = ny as usize * stride + nx as usize;
                let d1 = depth[there].to_f32();
                let dot: f32 = (0..3).map(|c| n0[c] * normal[c][there].to_f32()).sum();
                if (d1 - d0).abs() > 0.02 * d0.max(1e-3) || dot < 0.9 {
                    edge = true;
                }
            }
            mask[y * extent + x] = edge;
        }
    }
    mask
}

fn write_png(path: &str, rgb: &[f32], extent: usize, zoom: usize) -> std::io::Result<()> {
    let out = extent * zoom;
    let mut bytes = Vec::with_capacity(out * out * 4);
    for y in 0..out {
        for x in 0..out {
            let texel = &rgb[((y / zoom) * extent + x / zoom) * 3..];
            for &linear in &texel[..3] {
                bytes.push((display(linear) * 255.0 + 0.5) as u8);
            }
            bytes.push(255);
        }
    }
    let file = std::fs::File::create(path)?;
    let mut encoder = png::Encoder::new(file, out as u32, out as u32);
    encoder.set_color(png::ColorType::Rgba);
    encoder.set_depth(png::BitDepth::Eight);
    encoder
        .write_header()?
        .write_image_data(&bytes)
        .map_err(std::io::Error::other)
}

fn main() {
    let mut args = std::env::args().skip(1);
    let path = PathBuf::from(args.next().expect("usage: diagnose <dataset.omd> [crops]"));
    let limit: usize = args.next().map_or(128, |v| v.parse().unwrap());
    let dump = args.next();

    let mut reader = Reader::open(&path).expect("cannot open the dataset");
    let layout = *reader.layout();
    let guide = GuideConfig::TUNED;
    let tile = 64.min(layout.lr_width);
    let crops = Crop::grid(&layout, tile, tile);
    let extent = tile as usize * layout.scale as usize;
    println!(
        "{}\n  {} samples, {}x{} -> {}x{}, {} crops of {tile}, guide {guide:?}",
        path.display(),
        reader.len(),
        layout.lr_width,
        layout.lr_height,
        layout.hr_width(),
        layout.hr_height(),
        crops.len(),
    );

    let mut bilinear = Score::default();
    let mut shipped = Score::default();
    let mut upsample_only = Score::default();
    let mut edge = Score::default();
    let mut flat = Score::default();
    // What a model that only chose *where to filter how hard* could reach. The
    // candidates are the shipped filter at several footprints plus the raw
    // sample, selected per input texel by an oracle. It is an upper bound on
    // per-pixel kernel selection, not an achievable score, but unlike the
    // converged-colour oracle every candidate is something the runtime can
    // actually produce.
    const SIGMAS: [f32; 4] = [1.0, 2.0, 4.5, 9.0];
    // Selecting per texel could cheat, by picking whichever candidate's noise
    // happened to land near the truth. Deciding once per block cannot: the
    // choice has to be right for every texel under it. If the gain survives
    // coarsening, it is a real property of the image, not of this noise draw.
    const BLOCKS: [u32; 3] = [1, 4, 16];
    let mut adaptive = [Score::default(); BLOCKS.len()];
    let mut adaptive_detail = [0.0f64; BLOCKS.len()];
    // A gather with non-negative weights and non-negative taps cannot produce a
    // value below the smallest tap it read. Any output pixel that does is proof
    // that every weight in its footprint underflowed and `weight_sum.max(1e-12)`
    // returned the guard rather than a real normalisation.
    let mut collapsed_pixels = 0usize;
    let mut collapse_error = 0.0f64;
    let mut total_error_check = 0.0f64;
    let mut demodulated = Score::default();
    let mut demodulated_oracle = Score::default();
    let mut detail_demodulated = 0.0f64;
    let mut detail_demodulated_oracle = 0.0f64;
    let mut selection_counts = [0usize; SIGMAS.len() + 1];

    // Error share and pixel share by displayed reference luminance.
    const BANDS: [f32; 5] = [0.1, 0.25, 0.5, 0.75, 1.01];
    let mut band_error = [0.0f64; 5];
    let mut band_display_error = [0.0f64; 5];
    let mut band_pixels = [0usize; 5];

    let mut detail_reference = 0.0;
    let mut detail_shipped = 0.0;
    let mut detail_ideal = 0.0;
    let mut per_crop: Vec<(f64, usize, usize)> = Vec::new();
    let mut counted = 0usize;

    // Does the project's own metric prefer a blurrier frame?
    const RADII: [usize; 3] = [1, 2, 3];
    let mut blurred = [Score::default(); 3];
    let mut detail_blurred = [0.0f64; 3];
    // Smooth versus noisy error, at the scale the guide filters over.
    let mut error_smooth = 0.0f64;
    let mut error_noisy = 0.0f64;
    let mut reference_highpass = 0.0f64;
    let mut reference_values = 0usize;
    let mut ssim_block = 0.0f64;
    let mut ssim_gaussian = 0.0f64;
    let mut ssim_block_blurred = 0.0f64;
    let mut ssim_gaussian_blurred = 0.0f64;

    'outer: for index in 0..reader.len() {
        let sample = match reader.sample(index) {
            Ok(sample) => sample,
            Err(e) => {
                eprintln!("sample {index}: {e}");
                break;
            }
        };
        let oracle = oracle_low_resolution(&sample, &layout);
        let whole = Crop {
            x: 0,
            y: 0,
            tile: layout.lr_width,
        };
        // Candidate low-resolution colours, whole frame, one per footprint,
        // with the unfiltered sample as the last candidate.
        let mut candidates: Vec<Vec<f32>> = SIGMAS
            .iter()
            .map(|&spatial_sigma| {
                batch::guided_color(
                    &sample,
                    &layout,
                    whole,
                    GuideConfig {
                        spatial_sigma,
                        ..guide
                    },
                )
            })
            .collect();
        candidates.push(batch::crop_color(&sample, &layout, whole));
        let albedo_low = lr_albedo(&sample, &layout);
        // candidates[2] is spatial sigma 4.5, which is the shipped guide, so
        // this changes only how the result reaches output resolution.
        let demod_shipped = demodulate(&candidates[2], &albedo_low);
        let demod_oracle = demodulate(&oracle, &albedo_low);
        let width = layout.lr_width as usize;
        let texel_error = |candidate: &[f32], texel: usize| -> f32 {
            (0..3)
                .map(|c| {
                    let d = transform::compress(candidate[texel * 3 + c])
                        - transform::compress(oracle[texel * 3 + c]);
                    d * d
                })
                .sum()
        };
        let chosen: Vec<Vec<f32>> = BLOCKS
            .iter()
            .map(|&block| {
                let block = block as usize;
                let mut out = vec![0.0f32; oracle.len()];
                for by in (0..width).step_by(block) {
                    for bx in (0..width).step_by(block) {
                        let (best, _) = candidates
                            .iter()
                            .enumerate()
                            .map(|(i, candidate)| {
                                let mut error = 0.0;
                                for y in by..(by + block).min(width) {
                                    for x in bx..(bx + block).min(width) {
                                        error += texel_error(candidate, y * width + x);
                                    }
                                }
                                (i, error)
                            })
                            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                            .unwrap();
                        if block == 1 {
                            selection_counts[best] += 1;
                        }
                        for y in by..(by + block).min(width) {
                            for x in bx..(bx + block).min(width) {
                                let texel = y * width + x;
                                out[texel * 3..texel * 3 + 3]
                                    .copy_from_slice(&candidates[best][texel * 3..texel * 3 + 3]);
                            }
                        }
                    }
                }
                out
            })
            .collect();

        for (crop_index, &crop) in crops.iter().enumerate() {
            if counted >= limit {
                break 'outer;
            }
            let reference = batch::crop_reference(&sample, &layout, crop);
            let base = batch::high_resolution_guided_base(&sample, &layout, crop, guide);
            let ideal =
                batch::high_resolution_guided_from_color(&sample, &layout, crop, guide, &oracle);
            let low = batch::crop_color(&sample, &layout, crop);
            let up = bilinear_upsample(&low, tile as usize, layout.scale as usize);

            bilinear.accumulate(&up, &reference);
            shipped.accumulate(&base, &reference);
            upsample_only.accumulate(&ideal, &reference);

            {
                let width = layout.lr_width as usize;
                let scale = layout.scale as usize;
                let low = &candidates[2];
                for oy in 0..extent {
                    let global_y = crop.y as usize * scale + oy;
                    let py = (global_y as f32 + 0.5) / scale as f32 - 0.5;
                    for ox in 0..extent {
                        let global_x = crop.x as usize * scale + ox;
                        let px = (global_x as f32 + 0.5) / scale as f32 - 0.5;
                        let (bx, by) = (px.floor() as i32, py.floor() as i32);
                        let mut floor_value = [f32::MAX; 3];
                        for dy in -2..=2 {
                            let sy = (by + dy).clamp(0, layout.lr_height as i32 - 1) as usize;
                            for dx in -2..=2 {
                                let sx = (bx + dx).clamp(0, width as i32 - 1) as usize;
                                for c in 0..3 {
                                    floor_value[c] =
                                        floor_value[c].min(low[(sy * width + sx) * 3 + c]);
                                }
                            }
                        }
                        let pixel = oy * extent + ox;
                        let mut is_collapsed = false;
                        for c in 0..3 {
                            let got = base[pixel * 3 + c];
                            if got < floor_value[c] - 1e-4 {
                                is_collapsed = true;
                            }
                            let d = transform::compress(got)
                                - transform::compress(reference[pixel * 3 + c]);
                            total_error_check += (d * d) as f64;
                            if got < floor_value[c] - 1e-4 {
                                collapse_error += (d * d) as f64;
                            }
                        }
                        if is_collapsed {
                            collapsed_pixels += 1;
                        }
                    }
                }
            }

            let albedo_high = crop_hr_albedo(&sample, &layout, crop);
            for (target, tally, source) in [
                (&mut demodulated, &mut detail_demodulated, &demod_shipped),
                (
                    &mut demodulated_oracle,
                    &mut detail_demodulated_oracle,
                    &demod_oracle,
                ),
            ] {
                let up =
                    batch::high_resolution_guided_from_color(&sample, &layout, crop, guide, source);
                let out = remodulate(&up, &albedo_high);
                target.accumulate(&out, &reference);
                *tally += detail(&out, extent);
            }

            for (slot, low) in chosen.iter().enumerate() {
                let picked =
                    batch::high_resolution_guided_from_color(&sample, &layout, crop, guide, low);
                adaptive[slot].accumulate(&picked, &reference);
                adaptive_detail[slot] += detail(&picked, extent);
            }

            let mask = silhouette_mask(&sample, &layout, crop);
            for (pixel, &is_edge) in mask.iter().enumerate() {
                let a = &base[pixel * 3..pixel * 3 + 3];
                let b = &reference[pixel * 3..pixel * 3 + 3];
                if is_edge {
                    edge.accumulate(a, b);
                } else {
                    flat.accumulate(a, b);
                }
                let level = display(luminance(b));
                let band = BANDS.iter().position(|&t| level < t).unwrap_or(4);
                band_pixels[band] += 3;
                for c in 0..3 {
                    let dc = transform::compress(a[c]) - transform::compress(b[c]);
                    band_error[band] += (dc * dc) as f64;
                    let dd = display(a[c]) - display(b[c]);
                    band_display_error[band] += (dd * dd) as f64;
                }
            }

            detail_reference += detail(&reference, extent);
            detail_shipped += detail(&base, extent);
            detail_ideal += detail(&ideal, extent);
            ssim_block += ommatidia::metrics::ssim(&base, &reference, extent, extent) as f64;
            ssim_gaussian += gaussian_ssim(&base, &reference, extent);
            ssim_block_blurred +=
                ommatidia::metrics::ssim(&blur(&base, extent, 1), &reference, extent, extent)
                    as f64;
            ssim_gaussian_blurred += gaussian_ssim(&blur(&base, extent, 1), &reference, extent);

            for (slot, &radius) in RADII.iter().enumerate() {
                let soft = blur(&base, extent, radius);
                blurred[slot].accumulate(&soft, &reference);
                detail_blurred[slot] += detail(&soft, extent);
            }

            // Split the error itself, not the images: the smooth part is what a
            // better filter or a bigger model could predict, the rest is the
            // sample noise that only more samples can remove.
            let residual: Vec<f32> = base
                .iter()
                .zip(reference.iter())
                .map(|(&a, &b)| transform::compress(a) - transform::compress(b))
                .collect();
            let smooth = box_blur_raw(&residual, extent, 2);
            for (&r, &s) in residual.iter().zip(smooth.iter()) {
                error_smooth += (s * s) as f64;
                error_noisy += ((r - s) * (r - s)) as f64;
            }

            // How clean the canonical target itself is. High-passing it away
            // from silhouettes over-counts, because real shading detail lands
            // in the same band, so the implied PSNR is a floor rather than an
            // estimate. If that floor sits far above the reconstruction score,
            // the reference is not what is limiting the result.
            let compressed: Vec<f32> = reference.iter().map(|&x| transform::compress(x)).collect();
            let low = box_blur_raw(&compressed, extent, 1);
            for (pixel, &is_edge) in mask.iter().enumerate() {
                if is_edge {
                    continue;
                }
                for c in 0..3 {
                    let d = compressed[pixel * 3 + c] - low[pixel * 3 + c];
                    reference_highpass += (d * d) as f64;
                    reference_values += 1;
                }
            }

            let mut crop_score = Score::default();
            crop_score.accumulate(&base, &reference);
            per_crop.push((
                crop_score.displayed / crop_score.values as f64,
                index,
                crop_index,
            ));
            counted += 1;
        }
    }

    println!("\nreconstruction quality over {counted} crops");
    bilinear.report("texel-centre bilinear");
    shipped.report("shipped HR-guided base");
    for (slot, &block) in BLOCKS.iter().enumerate() {
        adaptive[slot].report(&format!("oracle footprint, {block}x{block}"));
        println!(
            "  {:<26} {:>62} detail {:5.1}%",
            "",
            "",
            100.0 * adaptive_detail[slot] / detail_reference,
        );
    }
    upsample_only.report("oracle colour + same guide");
    println!("\n  reconstructing in albedo-demodulated space instead:");
    demodulated.report("shipped colour, demodulated");
    println!(
        "  {:<26} {:>62} detail {:5.1}%",
        "",
        "",
        100.0 * detail_demodulated / detail_reference,
    );
    demodulated_oracle.report("oracle colour, demodulated");
    println!(
        "  {:<26} {:>62} detail {:5.1}%",
        "",
        "",
        100.0 * detail_demodulated_oracle / detail_reference,
    );
    println!(
        "  per-texel choice among sigma {:?} / raw: {:?}%",
        SIGMAS,
        selection_counts
            .iter()
            .map(
                |&c| (1000.0 * c as f64 / selection_counts.iter().sum::<usize>() as f64).round()
                    / 10.0
            )
            .collect::<Vec<_>>(),
    );

    println!("\nwhere the error is");
    edge.report("near a silhouette");
    flat.report("surface interior");
    println!(
        "  silhouette pixels are {:.1}% of the frame and carry {:.1}% of the compressed error, \
         {:.1}% of the displayed error",
        100.0 * edge.values as f64 / (edge.values + flat.values) as f64,
        100.0 * edge.compressed / (edge.compressed + flat.compressed),
        100.0 * edge.displayed / (edge.displayed + flat.displayed),
    );

    println!("\nerror by displayed reference luminance");
    println!("  band          pixels    project metric    displayed");
    let total: f64 = band_error.iter().sum();
    let total_display: f64 = band_display_error.iter().sum();
    let total_pixels: usize = band_pixels.iter().sum();
    let names = [
        "0.00-0.10",
        "0.10-0.25",
        "0.25-0.50",
        "0.50-0.75",
        "0.75-1.00",
    ];
    for i in 0..5 {
        println!(
            "  {:<10} {:7.1}%          {:6.1}%      {:6.1}%",
            names[i],
            100.0 * band_pixels[i] as f64 / total_pixels as f64,
            100.0 * band_error[i] / total,
            100.0 * band_display_error[i] / total_display,
        );
    }

    println!("\ndetail retained");
    println!(
        "  mean displayed gradient: reference {:.5}, reconstruction {:.5} ({:.1}% retained)\n  \
         the same upsampler fed perfect colour retains {:.1}%, so {:.1} of the {:.1} points \
         lost are the denoiser rather than the 2x reconstruction",
        detail_reference / counted as f64,
        detail_shipped / counted as f64,
        100.0 * detail_shipped / detail_reference,
        100.0 * detail_ideal / detail_reference,
        100.0 * (detail_ideal - detail_shipped) / detail_reference,
        100.0 * (detail_reference - detail_shipped) / detail_reference,
    );

    println!("\nwhat happens if the frame is blurred further");
    println!(
        "  {:<26} PSNR {:6.2} dB                            detail {:5.1}%",
        "unblurred",
        psnr(shipped.compressed / shipped.values as f64),
        100.0 * detail_shipped / detail_reference,
    );
    for (slot, &radius) in RADII.iter().enumerate() {
        blurred[slot].report(&format!("box blur radius {radius}"));
        println!(
            "  {:<26} {:>62} detail {:5.1}%",
            "",
            "",
            100.0 * detail_blurred[slot] / detail_reference,
        );
    }

    println!("\nSSIM, project 8x8 blocks versus the standard Gaussian window");
    println!(
        "  as shipped         blocks {:.4}   gaussian {:.4}\n  \
         plus a radius-1 blur blocks {:.4}   gaussian {:.4}",
        ssim_block / counted as f64,
        ssim_gaussian / counted as f64,
        ssim_block_blurred / counted as f64,
        ssim_gaussian_blurred / counted as f64,
    );

    println!(
        "\nhow clean the canonical target is\n  \
         high-pass energy away from silhouettes implies at worst {:.1} dB, \
         against a {:.2} dB reconstruction",
        psnr(reference_highpass / reference_values as f64),
        psnr(shipped.compressed / shipped.values as f64),
    );

    println!(
        "\nweight collapse in the output-resolution gather\n  \
         {collapsed_pixels} pixels ({:.2}% of the frame) fall below every tap they read, \
         carrying {:.1}% of the total error",
        100.0 * collapsed_pixels as f64 / (counted * extent * extent) as f64,
        100.0 * collapse_error / total_error_check,
    );

    println!("\nerror composition");
    let total_error = error_smooth + error_noisy;
    println!(
        "  smooth (a filter or a model could predict it) {:5.1}%\n  \
         per-pixel noise (only more samples remove it)   {:5.1}%",
        100.0 * error_smooth / total_error,
        100.0 * error_noisy / total_error,
    );

    per_crop.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    println!("\nper-crop displayed PSNR");
    let values: Vec<f64> = per_crop.iter().map(|c| psnr(c.0)).collect();
    println!(
        "  worst {:.2} dB   p10 {:.2}   median {:.2}   best {:.2} dB",
        values[0],
        values[values.len() / 10],
        values[values.len() / 2],
        values[values.len() - 1],
    );
    let worst_share: f64 = per_crop[..per_crop.len() / 20]
        .iter()
        .map(|c| c.0)
        .sum::<f64>()
        / per_crop.iter().map(|c| c.0).sum::<f64>();
    println!(
        "  the worst 5% of crops carry {:.1}% of the displayed error",
        100.0 * worst_share
    );

    if let Some(prefix) = dump {
        let (_, index, crop_index) = per_crop[0];
        let sample = reader.sample(index).unwrap();
        let crop = crops[crop_index];
        let reference = batch::crop_reference(&sample, &layout, crop);
        let base = batch::high_resolution_guided_base(&sample, &layout, crop, guide);
        let low = batch::crop_color(&sample, &layout, crop);
        let up = bilinear_upsample(&low, tile as usize, layout.scale as usize);
        for (name, image) in [
            ("reference", &reference),
            ("shipped", &base),
            ("input", &up),
        ] {
            let path = format!("{prefix}-{name}.png");
            write_png(&path, image, extent, 3).unwrap();
            println!("wrote {path}");
        }
    }
}

/// Separable box blur of a plain signed signal, with no colour transform.
fn box_blur_raw(image: &[f32], extent: usize, radius: usize) -> Vec<f32> {
    let mut mid = vec![0.0f32; image.len()];
    let mut out = vec![0.0f32; image.len()];
    for y in 0..extent {
        for x in 0..extent {
            for c in 0..3 {
                let mut sum = 0.0;
                let mut count = 0.0;
                for dx in -(radius as i32)..=radius as i32 {
                    let sx = (x as i32 + dx).clamp(0, extent as i32 - 1) as usize;
                    sum += image[(y * extent + sx) * 3 + c];
                    count += 1.0;
                }
                mid[(y * extent + x) * 3 + c] = sum / count;
            }
        }
    }
    for y in 0..extent {
        for x in 0..extent {
            for c in 0..3 {
                let mut sum = 0.0;
                let mut count = 0.0;
                for dy in -(radius as i32)..=radius as i32 {
                    let sy = (y as i32 + dy).clamp(0, extent as i32 - 1) as usize;
                    sum += mid[(sy * extent + x) * 3 + c];
                    count += 1.0;
                }
                out[(y * extent + x) * 3 + c] = sum / count;
            }
        }
    }
    out
}

fn bilinear_upsample(low: &[f32], width: usize, scale: usize) -> Vec<f32> {
    let out_width = width * scale;
    let mut out = vec![0.0; out_width * out_width * 3];
    for oy in 0..out_width {
        let fy = (oy as f32 + 0.5) / scale as f32 - 0.5;
        let y0 = fy.floor() as isize;
        let ty = fy - y0 as f32;
        let y0c = y0.clamp(0, width as isize - 1) as usize;
        let y1c = (y0 + 1).clamp(0, width as isize - 1) as usize;
        for ox in 0..out_width {
            let fx = (ox as f32 + 0.5) / scale as f32 - 0.5;
            let x0 = fx.floor() as isize;
            let tx = fx - x0 as f32;
            let x0c = x0.clamp(0, width as isize - 1) as usize;
            let x1c = (x0 + 1).clamp(0, width as isize - 1) as usize;
            for c in 0..3 {
                let p00 = low[(y0c * width + x0c) * 3 + c];
                let p10 = low[(y0c * width + x1c) * 3 + c];
                let p01 = low[(y1c * width + x0c) * 3 + c];
                let p11 = low[(y1c * width + x1c) * 3 + c];
                let top = p00 + tx * (p10 - p00);
                let bottom = p01 + tx * (p11 - p01);
                out[(oy * out_width + ox) * 3 + c] = top + ty * (bottom - top);
            }
        }
    }
    out
}
