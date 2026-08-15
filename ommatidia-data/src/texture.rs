//! Procedural base-colour textures, baked to PNG in memory.
//!
//! `ProceduralGeometry` carries a colour factor and nothing else, so every
//! surface the generator built was one flat colour. That is why the validation
//! data cannot discriminate a good reconstruction from a mediocre one: within
//! an object the ground truth is a smooth low-frequency function that the
//! G-buffer already segments exactly, so a bilateral filter is near optimal by
//! construction and there is almost nothing for a network to recover. Measured
//! on that data, albedo demodulation — one of the larger wins available in a
//! production denoiser — changes the score by 0.01 dB.
//!
//! These go through the same asset path as a glTF model's, BC1 and all, so what
//! the renderer samples here is what it would sample from a real material. The
//! patterns are greyscale; each surface's own colour factor tints them, which
//! keeps the palette small while leaving every object a different colour.

use ommatidia::rng::Rng;

/// Edge length of a baked texture. Large enough that the finest features land
/// well below one input pixel at the distances these scenes are viewed from.
pub const SIZE: usize = 256;

/// What a baked texture looks like.
///
/// Three shapes rather than one, because they fail differently: a filter that
/// copes with gradual variation can still erase isolated features, and one that
/// keeps isolated features can still round off a hard edge.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// Hard-edged squares. The highest-contrast albedo edge available, at a
    /// frequency the input resolution cannot represent.
    Checker,
    /// Band-limited value noise over several octaves, which is closer to what
    /// a real surface carries and has energy at every scale rather than one.
    Noise,
    /// Small dark discs on a light field. What a reconstruction that averages
    /// too eagerly removes completely, leaving nothing to notice.
    Dots,
}

pub const KINDS: [Kind; 3] = [Kind::Checker, Kind::Noise, Kind::Dots];

/// Value noise on a `lattice`-spaced grid, bilinearly interpolated and wrapped
/// so the texture tiles without a seam.
fn value_noise(lattice: usize, rng: &mut Rng) -> Vec<f32> {
    let corners: Vec<f32> = (0..lattice * lattice).map(|_| rng.uniform()).collect();
    let at = |x: usize, y: usize| corners[(y % lattice) * lattice + (x % lattice)];
    let step = SIZE as f32 / lattice as f32;
    let mut out = vec![0.0; SIZE * SIZE];
    for y in 0..SIZE {
        for x in 0..SIZE {
            let (fx, fy) = (x as f32 / step, y as f32 / step);
            let (x0, y0) = (fx.floor() as usize, fy.floor() as usize);
            let (tx, ty) = (fx - x0 as f32, fy - y0 as f32);
            // Smoothstep, so the lattice does not show as a grid of creases.
            let (sx, sy) = (tx * tx * (3.0 - 2.0 * tx), ty * ty * (3.0 - 2.0 * ty));
            let top = at(x0, y0) + sx * (at(x0 + 1, y0) - at(x0, y0));
            let bottom = at(x0, y0 + 1) + sx * (at(x0 + 1, y0 + 1) - at(x0, y0 + 1));
            out[y * SIZE + x] = top + sy * (bottom - top);
        }
    }
    out
}

/// Greyscale coverage in `[0, 1]` for one texture, before encoding.
fn pattern(kind: Kind, seed: u64) -> Vec<f32> {
    let mut rng = Rng::new(seed);
    match kind {
        Kind::Checker => {
            let cell = 16 + 16 * (rng.uniform() * 2.0) as usize;
            // Not quite two-tone: each square is shaded a little differently,
            // so the pattern cannot be reconstructed from two learned values.
            let shades: Vec<f32> = (0..(SIZE / cell + 1).pow(2))
                .map(|_| 0.35 + 0.6 * rng.uniform())
                .collect();
            let columns = SIZE / cell + 1;
            (0..SIZE * SIZE)
                .map(|index| {
                    let (x, y) = (index % SIZE / cell, index / SIZE / cell);
                    let base = shades[y * columns + x];
                    if (x + y).is_multiple_of(2) {
                        base
                    } else {
                        base * 0.45
                    }
                })
                .collect()
        }
        Kind::Noise => {
            let mut out = vec![0.0f32; SIZE * SIZE];
            let mut amplitude = 1.0;
            let mut total = 0.0;
            for octave in 0..4 {
                let lattice = 4 << octave;
                let layer = value_noise(lattice, &mut rng);
                for (value, &noise) in out.iter_mut().zip(layer.iter()) {
                    *value += amplitude * noise;
                }
                total += amplitude;
                amplitude *= 0.55;
            }
            out.iter_mut().for_each(|value| {
                *value = 0.35 + 0.6 * (*value / total);
            });
            out
        }
        Kind::Dots => {
            let mut out = vec![1.0f32; SIZE * SIZE];
            let cell = 24;
            let cells = SIZE / cell;
            for row in 0..cells {
                for column in 0..cells {
                    let radius = cell as f32 * (0.12 + 0.2 * rng.uniform());
                    let shade = 0.3 + 0.25 * rng.uniform();
                    let center = (
                        (column as f32 + 0.15 + 0.7 * rng.uniform()) * cell as f32,
                        (row as f32 + 0.15 + 0.7 * rng.uniform()) * cell as f32,
                    );
                    let span = radius.ceil() as i32 + 1;
                    for dy in -span..=span {
                        for dx in -span..=span {
                            let x = center.0 as i32 + dx;
                            let y = center.1 as i32 + dy;
                            if x < 0 || y < 0 || x >= SIZE as i32 || y >= SIZE as i32 {
                                continue;
                            }
                            let distance = ((x as f32 - center.0).powi(2)
                                + (y as f32 - center.1).powi(2))
                            .sqrt();
                            if distance <= radius {
                                out[y as usize * SIZE + x as usize] = shade;
                            }
                        }
                    }
                }
            }
            out
        }
    }
}

/// Bake one texture and encode it as PNG.
///
/// PNG rather than raw because that is what the asset cooker decodes, and going
/// through it means these are compressed to BC1 and mipped exactly as a glTF
/// material's base colour would be.
pub fn bake(kind: Kind, seed: u64) -> Vec<u8> {
    let coverage = pattern(kind, seed);
    // The pattern is a reflectance, and the texture is sampled as sRGB.
    let encode = |linear: f32| {
        let value = if linear <= 0.003_130_8 {
            12.92 * linear
        } else {
            1.055 * linear.powf(1.0 / 2.4) - 0.055
        };
        (value.clamp(0.0, 1.0) * 255.0 + 0.5) as u8
    };
    let mut bytes = Vec::with_capacity(SIZE * SIZE * 4);
    for &value in &coverage {
        let encoded = encode(value);
        bytes.extend_from_slice(&[encoded, encoded, encoded, 255]);
    }

    let mut png = Vec::new();
    {
        let mut encoder = png::Encoder::new(&mut png, SIZE as u32, SIZE as u32);
        encoder.set_color(png::ColorType::Rgba);
        encoder.set_depth(png::BitDepth::Eight);
        let mut writer = encoder.write_header().expect("png header");
        writer.write_image_data(&bytes).expect("png data");
    }
    png
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_pattern_covers_its_range_without_leaving_it() {
        for kind in KINDS {
            let coverage = pattern(kind, 7);
            assert_eq!(coverage.len(), SIZE * SIZE);
            let low = coverage.iter().cloned().fold(f32::MAX, f32::min);
            let high = coverage.iter().cloned().fold(f32::MIN, f32::max);
            assert!(
                (0.0..=1.0).contains(&low) && (0.0..=1.0).contains(&high),
                "{kind:?} left the unit range: {low} to {high}"
            );
            // A texture that multiplies the colour factor by a nearly constant
            // value is a texture that is not doing anything.
            assert!(
                high - low > 0.25,
                "{kind:?} spans only {low} to {high}, which is no pattern at all"
            );
        }
    }

    #[test]
    fn baking_produces_a_decodable_png_of_the_right_size() {
        for kind in KINDS {
            let bytes = bake(kind, 3);
            let decoder = png::Decoder::new(std::io::Cursor::new(&bytes));
            let reader = decoder.read_info().expect("decodes");
            let info = reader.info();
            assert_eq!((info.width as usize, info.height as usize), (SIZE, SIZE));
        }
    }

    /// Two seeds have to give two textures, or the palette is one texture.
    #[test]
    fn seeds_produce_different_patterns() {
        for kind in KINDS {
            let a = pattern(kind, 1);
            let b = pattern(kind, 2);
            let differing = a.iter().zip(b.iter()).filter(|(x, y)| x != y).count();
            assert!(
                differing > a.len() / 10,
                "{kind:?} ignored its seed: only {differing} of {} texels differ",
                a.len()
            );
        }
    }
}
