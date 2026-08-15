//! The encoding between stored physical quantities and network tensors.
//!
//! The dataset holds what the renderer produced: linear radiance, view-space
//! distance, unit normals. The network wants bounded, well-scaled inputs. This
//! module is the one place that conversion is defined, and it is mirrored by
//! `shaders/pack.wgsl` and `shaders/unpack.wgsl` so the trainer and the
//! runtime agree exactly.
//!
//! Note this is deliberately *not* part of the file format. Storing radiance
//! range-compressed in `f16` would be a mistake: `f16` already spends its bits
//! on an exponent, giving roughly 0.1% relative precision across its whole
//! range, which is exactly what high dynamic range wants. Compressing first
//! pushes every bright value up against 1.0, where `f16` steps by 1/2048 —
//! a radiance of 1000 and a radiance of 2000 would land on adjacent
//! representable values. So: store raw, transform on load.

/// Compress unbounded linear radiance into `[0, 1)`.
///
/// `x / (1 + x)` is monotonic, invertible, and maps the whole positive real
/// line into the unit interval, so nothing has to be clipped and the network
/// trains against a bounded target. Negative input is clamped — radiance is
/// non-negative, and a negative value would otherwise cross the pole at
/// `x = -1`.
pub fn compress(x: f32) -> f32 {
    let x = x.max(0.0);
    x / (1.0 + x)
}

/// Inverse of [`compress`].
///
/// Values at the very top of the range are held off 1.0, where the inverse
/// diverges.
pub fn decompress(y: f32) -> f32 {
    // 1 - 2^-12 caps the result around 4095, well inside f16 and far above any
    // radiance that survives tone mapping.
    let y = y.clamp(0.0, 1.0 - 1.0 / 4096.0);
    y / (1.0 - y)
}

/// Compress and sRGB-encode: the value that reaches the screen.
///
/// [`compress`] alone is nearly linear near zero, so a metric computed in it
/// weights a dark pixel by roughly its radiance rather than by how visible an
/// error there would be. This is the transform the evaluator's PNGs already
/// apply, so scoring in it scores the image that was looked at.
pub fn display(x: f32) -> f32 {
    let mapped = compress(x);
    let encoded = if mapped <= 0.003_130_8 {
        12.92 * mapped
    } else {
        1.055 * mapped.powf(1.0 / 2.4) - 0.055
    };
    encoded.clamp(0.0, 1.0)
}

/// Map view-space distance to `(0, 1]`.
///
/// Raw distance is unbounded and its useful precision is concentrated near the
/// camera, which is exactly what inverse depth expresses.
pub fn encode_depth(d: f32) -> f32 {
    1.0 / (1.0 + d.max(0.0))
}

/// Inverse of [`encode_depth`].
pub fn decode_depth(e: f32) -> f32 {
    let e = e.clamp(1.0 / 4096.0, 1.0);
    1.0 / e - 1.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use half::f16;

    #[test]
    fn compression_roundtrips_across_the_range() {
        for &x in &[0.0f32, 0.5, 1.0, 12.5, 100.0, 1000.0] {
            let y = compress(x);
            assert!((0.0..1.0).contains(&y), "{x} compressed to {y}");
            let back = decompress(y);
            assert!(
                (back - x).abs() <= x * 1e-3 + 1e-6,
                "{x} roundtripped to {back}"
            );
        }
        assert_eq!(compress(-1.0), 0.0, "negative radiance is clamped");
        // The clamp keeps the inverse finite at the top of the range.
        assert!(decompress(1.0).is_finite());
    }

    #[test]
    fn raw_f16_beats_compressed_f16_on_bright_values() {
        // The reason compression is a load-time transform and not a storage
        // format. Both paths store one f16; only the order differs.
        for &x in &[100.0f32, 1000.0, 10000.0] {
            let raw = f16::from_f32(x).to_f32();
            let compressed = decompress(f16::from_f32(compress(x)).to_f32());

            let raw_error = (raw - x).abs() / x;
            let compressed_error = (compressed - x).abs() / x;
            assert!(raw_error < 0.001, "raw f16 lost {raw_error} of {x} ({raw})");
            assert!(
                compressed_error > raw_error * 10.0,
                "expected compressed storage to be far worse at {x}: \
                 raw {raw_error} vs compressed {compressed_error}"
            );
        }
    }

    #[test]
    fn depth_encoding_roundtrips() {
        for &d in &[0.0f32, 0.1, 1.0, 50.0, 500.0] {
            let e = encode_depth(d);
            assert!((0.0..=1.0).contains(&e), "{d} encoded to {e}");
            let back = decode_depth(e);
            assert!((back - d).abs() <= d * 1e-3 + 1e-6, "{d} became {back}");
        }
        // Nearer geometry gets more of the range, which is the point.
        assert!(encode_depth(0.5) - encode_depth(1.0) > encode_depth(10.0) - encode_depth(20.0));
    }
}
