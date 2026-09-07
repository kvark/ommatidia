from pathlib import Path

p = Path('ommatidia/src/batch.rs')
s = p.read_text()
a = s.index('/// Versioned previous-output warp.')
z = s.index('/// Convert an interleaved linear crop', a)
s = s[:a] + '''/// Replay the native previous-output texture contract, including f16 storage.
///
/// Legacy checkpoints interpolate compressed RGB. Physical modes decode before
/// storing and interpolate linear radiance. The temporal-loss helper above
/// deliberately interpolates linear radiance in every mode to match its metric;
/// it is not interchangeable with replay of a legacy runtime history texture.
pub fn warp_previous_output_for_fusion(
    previous: &[f32],
    warp: crate::temporal::Reprojection<'_>,
    tile: usize,
    scale: usize,
    mode: crate::fusion::Mode,
) -> WarpedOutput {
    let extent = tile * scale;
    let slots = scale * scale;
    assert_eq!(previous.len(), 3 * extent * extent);
    assert_eq!(warp.current.len(), extent * extent);
    assert_eq!(warp.previous.len(), extent * extent);
    let stored: Vec<_> = spread(previous, tile, scale).into_iter().map(|value| {
        let value = if mode.is_linear() { transform::decompress(value) } else { value };
        f16::from_f32(value).to_f32()
    }).collect();
    let mut color = vec![0.0; previous.len()];
    let mut validity = vec![0.0; slots * tile * tile];
    for y in 0..extent {
        for x in 0..extent {
            let motion = warp.output_motion(x, y, [tile, tile], scale);
            let position = [x as f32 + motion[0], y as f32 + motion[1]];
            let Some(rgb) = crate::temporal::sample_reprojected(
                &stored, warp.previous, warp.current[y * extent + x], position,
                extent, extent, warp.rejection,
            ) else { continue };
            let slot = (y % scale) * scale + x % scale;
            let pixel = (y / scale) * tile + x / scale;
            validity[slot * tile * tile + pixel] = 1.0;
            for c in 0..3 {
                color[(c * slots + slot) * tile * tile + pixel] =
                    if mode.is_linear() { transform::compress(rgb[c]) } else { rgb[c] };
            }
        }
    }
    WarpedOutput { color, validity }
}

''' + s[z:]
s = s.replace('let expected = if mode.is_linear() { 2.0 } else { 2.0 / 3.0 };', '''let expected = if mode.is_linear() {
                2.0
            } else {
                // The native legacy texture quantizes compressed values.
                transform::decompress(0.5 * f16::from_f32(transform::compress(4.0)).to_f32())
            };''')
s = s.replace('assert!((transform::decompress(warped.color[0]) - expected).abs() < 1.0e-5);', 'assert!((transform::decompress(warped.color[0]) - expected).abs() < 1.0e-5, "{mode:?}: {} != {expected}", transform::decompress(warped.color[0]));')
p.write_text(s)

p = Path('ommatidia/tests/gpu_runtime.rs')
s = p.read_text()
a = s.index('fn candidate_local_training_updates_finite_parameters()')
s = s[:a] + s[a:].replace('base_channels: 8,', 'base_channels: 8,\n        temporal_weight: 0.0,', 1)
p.write_text(s)

p = Path('README.md')
s = p.read_text().replace('exist only for matched Blade baselines', 'exist only for historical Blade pipeline controls')
p.write_text(s)

p = Path('docs/reconstruction-review-2026-09-06.md')
s = p.read_text().replace('History reprojection also interpolates compressed pixels, and history is stored compressed in f16.', 'Native history reprojection also interpolates compressed pixels, and history is stored compressed in f16. The old CPU history helper instead interpolates linear radiance, so fractional-motion CPU replay already differed from the native legacy path.')
s = s.replace('CPU teacher/evaluation replay quantizes that linear history to match native storage.', 'CPU teacher/evaluation replay now explicitly reproduces each mode\'s native storage and interpolation: compressed f16 for Legacy, linear f16 for the new modes. The separate reference-change temporal-loss resampler stays linear. This also corrects the old fractional-motion CPU/native replay discrepancy without changing legacy native checkpoint behavior.')
s = s.replace('| A | legacy | group-norm | matched-data baseline |', '| A | legacy | group-norm | matched-data, native-consistent replay baseline |')
p.write_text(s)

# git diff --check treats extra blank lines at EOF as an error.
for p in [Path('README.md'), *Path('docs').glob('*.md')]:
    p.write_text(p.read_text().rstrip() + '\n')
