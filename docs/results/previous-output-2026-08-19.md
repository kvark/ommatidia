# Previous reconstructed frame as history, 2026-08-19

## Why this, not another rejection tweak

DLSS-like quality is not “a slightly better mask on sparse samples.” A
temporal upscaler reuses the *picture it already made*, warped by
motion, and only invents pixels the warp cannot explain. Ommatidia's
history tap was the opposite: gated accumulation of the same noisy
sparse samples the current frame already has. Two experiments on that
path — teacher occlusion, then an ungated second sample tap — moved
nothing the eye would call quality.

The speed gap is separate and larger. Kernel b16 is 14.4 ms / 61 GFLOP
at 1080p on a 7900 XT. DLSS-class is about 1–2 ms. That factor of
eight is not going to come from a 580-parameter tap. It comes from a
smaller network once history is doing real work.

## The change

`--previous-output` replaces the accumulated-sample history tap with
four taps, one per output sub-pixel, holding the previous
reconstruction warped by the teacher's occlusion-aware reprojection.
A history reset (first frame of a pair, or a missing warp) copies the
current sample so the tap is a no-op rather than black.

Training still unrolls two frames: the teacher reconstructs t−1 from
spatial samples only (reset history), and t gathers the warped
teacher picture. Evaluation is fully recurrent across the sequence.

Old sidecars keep a single accumulated-sample tap. 116 output
channels instead of 104, 306,020 parameters, 63.8 GFLOP.

## Result

Same 8,000-step b16 kernel, demodulated, temporal weight 1, same data
and seed. External validation, 1,524 crops. Previous-output evaluation
seeds frame 0 so it scores 1,475 temporal pairs instead of 982; the
HR-guide temporal MSE is therefore a slightly different set. The
comparison that matters is still network versus that guide.

| | PSNR | SSIM | relMSE | worst crop | detail | temporal | moving |
|---|---:|---:|---:|---:|---:|---:|---:|
| four-frame HR guide | 28.10 dB | 0.8664 | 0.203 | 8.17 | 44% | — | — |
| kernel, sample history (`tl-teacher`) | **29.56 dB** | **0.8745** | 0.0745 | 14.14 | 78% | +0.03 | +0.14 |
| **kernel, previous output** | 29.36 dB | 0.8682 | **0.0694** | **11.11** | **84%** | **+0.53** | **+0.22** |

Temporal stability moves half a decibel, the first time that column
has moved since the temporal loss itself. Relative error, worst crop,
and detail all improve. PSNR pays 0.20 dB — the usual tax for agreeing
with the past.

On the training-file holdout the same flip is larger because the
scenes are easier: temporal goes from −0.75 dB to +0.67 dB against the
HR guide.

## What this does not settle

**This is still not DLSS quality.** 29.4 dB on these 4-spp rich scenes,
with moving pixels only 0.22 dB above the deterministic guide, is a
better reconstruction, not a finished one. 1-spp (the budget an
upscaler actually gets) has not been re-measured under this tap.

**This is still not DLSS speed.** 64 GFLOP / 15 ms at 1080p is
semi-realtime. The width cut this result unlocked is recorded in
[`previous-output-b8-2026-08-19.md`](previous-output-b8-2026-08-19.md):
b8 is 9.1 ms and 3 dB worse, and recurrence makes the grain worse.
Do not drop to b8. Mixing previous-output *after* the gather, rather
than as extra taps, is in
[`previous-output-mix-2026-08-20.md`](previous-output-mix-2026-08-20.md):
14.66 ms at matched quality.
