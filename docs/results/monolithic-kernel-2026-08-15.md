# One reconstruction instead of two, 2026-08-15

## Question

The published checkpoint's learned residual adds 0.02 dB over the deterministic
filter it corrects, and the reconstruction is visibly soft. Is that the ceiling
of single-frame spatial reconstruction, as
[`the guided result`](path-trace-guided-2026-08-13.md) concluded, or the ceiling
of how the problem was posed?

## What the reported score could not see

`ommatidia/examples/diagnose.rs` scores the deterministic reconstruction on the
CPU and reproduces the tuned guide's 34.72 dB to within 0.04, then asks four
things PSNR and SSIM do not.

**Detail.** Displayed gradient energy, as a fraction of the canonical frame's.
The shipped HR guide keeps 63%.

**Whether the metrics penalise blur.** They do not:

| 4-spp validation, 128 crops | PSNR | display PSNR | SSIM | detail |
|---|---:|---:|---:|---:|
| shipped HR-guided base | 34.68 dB | 35.15 dB | 0.9574 | 64.0% |
| the same, plus a radius-1 box blur | 34.62 dB | **35.40 dB** | 0.9553 | 56.5% |

Another eighth of the frame's detail costs 0.06 dB and 0.002 SSIM, and
*improves* the same error measured in display space. The 8×8-block SSIM was
checked against a standard 11×11 Gaussian-window SSIM in case the blocking
flattered it; it does not (0.9574 against 0.9652). Both are simply blind here.

**Where the error is.** Silhouettes are 27.4% of pixels and 47.5% of the error.
59.4% of the residual error at 4 spp — 73.9% at 1 spp — is spatially
structured rather than per-pixel noise, which contradicts the premise that what
remains after the guide is mostly unobserved Monte Carlo noise.

**A defect.** 144 pixels, 0.01% of the frame, came out *below every tap they
read*, which a weighted average of non-negative taps cannot do.
`guide_similarity` returns exactly zero for a tap whose normal faces away and
again for a background centre beside geometry, so at a silhouette all 25 taps
can be rejected together and the normalisation divides by its floor. They
carried 3.5% of the error, about 140 per 1080p frame, and they flicker. Falling
back to the guide-free gather is worth 0.15 dB, and 0.42 dB in display space —
the gap between those two being the point.

## How much was available

An oracle chooses, per input texel, among the filter footprints the runtime
already ships: the same guide at spatial sigma 1, 2, 4.5 and 9, plus the
unfiltered sample. A per-texel oracle can cheat by picking whichever
candidate's noise landed near the truth, so the choice is also forced constant
over 4×4 and 16×16 blocks.

| | 4 spp | 1 spp |
|---|---:|---:|
| shipped | 34.68 | 32.12 |
| oracle, per texel | 36.91 | 35.45 |
| oracle, constant per 4×4 | 35.51 | 33.14 |
| oracle, constant per 16×16 | 34.95 | 32.39 |
| perfect colour, same guide | 38.12 | 38.12 |

Reading the 4×4 row as a floor and the per-texel row as a ceiling, per-pixel
filter adaptation is worth **+0.83 to +2.23 dB at 4 spp** and **+1.02 to
+3.33 dB at 1 spp**. The tuned global sigma of 4.5 is the best choice for 14.7%
of texels; 29.9% want 1.0, 29.5% want 9.0, and 9.5% want no filter at all.

The last row is the other half of the finding. Feeding the *converged*
reference through the existing 5×5 gather scores 38.12 dB and retains 61.5% of
the reference's detail — less than the shipped reconstruction's 64.0%, whose
excess is residual noise counted as gradient. So none of the softness was the
denoiser: the upsampler set it, and no improvement to denoising could move it.
The reference is not the limit either; its own high-frequency energy away from
silhouettes implies at worst 46.6 dB.

## The change

`Prediction::SubpixelKernel` makes denoising and upscaling one operation. The
network emits, per output sub-pixel, a weight for each input sample in a
neighbourhood, and the output is their normalised sum. There is no filtered
low-resolution image in the middle and no deterministic base to correct, which
`ReconstructionBase::Sample` names.

Predicting weights rather than colour is what matters. A residual over a filter
asks a least-squares network to predict that filter's error, which is dominated
by the noise the renderer happened to draw and whose conditional mean is very
nearly zero. Weights are a choice among samples the network can see, and there
is no degenerate answer. Two properties then follow from the form rather than
from training: the output is a convex combination of measured radiance, so it
cannot overshoot, invent energy, or go black; and it is one pass over one
neighbourhood, so nothing is filtered twice.

Mechanically, the gather has to live inside the graph, because a kernel that is
never applied has no gradient. Reducing over the tap axis is the one shape
meganeura has no primitive for, so it is a 1×1 convolution against constant
ones — which keeps the whole gather near sixty operations rather than the
thousand a per-channel decomposition needs. Softplus gives positivity that
cannot overflow and is zero nowhere, so every tap keeps a gradient; its inverse
is closed form, so the head bias starts the untrained network at texel-centre
bilinear. Only training builds the gather: at runtime the weights are the
output and `unpack.wgsl` applies them in one dispatch.

## Result

Trained on the 2,400-scene 4-spp HR-G-buffer set, 8,000 steps, Adam with cosine
decay from 3e-4 to 1e-5, radius 2 (25 taps, 100 output channels). Scored on the
separate seed-10000 validation set, 128 crops:

| reconstruction | PSNR | SSIM | relMSE | detail |
|---|---:|---:|---:|---:|
| texel-centre bilinear | 26.51 dB | 0.5776 | 0.08941 | 394% |
| low-resolution guide | 34.03 dB | 0.9508 | 0.02058 | 61% |
| HR guide 5×5 | 34.87 dB | 0.9579 | 0.01043 | 63% |
| kernel b8 r2 | 35.38 dB | 0.9338 | 0.00815 | 99% |
| **kernel b16 r2** | **36.61 dB** | 0.9514 | **0.00595** | **86%** |

Against the HR guide the b16 kernel is **+1.74 dB with 43% less relative error
and 86% of the reference's detail rather than 63%**, and against the previously
published 34.74 dB it is +1.87 dB. Both arms land inside the oracle bracket
predicted above, which is the strongest evidence that the bracket measured the
right thing.

Detail above 100% is noise, not sharpness, and the b8 arm sits at 99% with SSIM
*below* the guide's: it under-smooths, and the frame reads grainy where the old
one read blurry. The b16 arm resolves that — 86% detail with SSIM back to
0.9514 — which is the failure mode being fixed by capacity rather than traded
away.

**Capacity now matters, and it did not before.** The previous result found a
649k-parameter model matching a 74k one and closed the question. Under kernel
prediction, b8 → b16 is worth +1.23 dB on the same data. What saturated was the
parameterisation, not the backbone.

## Cost

Network only, 960×540 → 1920×1080, on a Radeon RX 7900 XT whose load counter
was unavailable, so these are not idle-validated:

| | ms | GFLOP | output channels |
|---|---:|---:|---:|
| residual b8 | 7.18 | 12.9 | 12 |
| kernel b8 r2 | 8.52 | 19.4 | 100 |
| kernel b16 r2 | 14.40 | 60.7 | 100 |

The fixed stages move the other way: the kernel path drops the 13×13 guided
filter from pack and the 25-tap bilateral from unpack, and adds a 25-tap gather.
The head is a 3×3 convolution to 100 channels and is now over a third of the
b8 arithmetic; a 1×1 head is the obvious thing to measure next.

## On scenes that carry shadow and texture

The section above originally ended by saying the scenes could not discriminate
these results, and predicting that the kernel formulation would gain more on
harder content rather than less. `--canopy`, `--textures` and `--gloss` make
that measurable. A matched 2,400-scene training set and a separate 128-scene
seed-10000 validation set were generated with all three, and both architectures
were retrained on them for the same 8,000 steps.

The new set is a different problem: 35.6% of its pixels fall below a displayed
luminance of 0.10 where none did before, it carries 2.06× the gradient energy,
and texel-centre bilinear scores 21.34 dB rather than 26.51.

| external validation, 128 crops | PSNR | SSIM | relMSE | worst crop | detail |
|---|---:|---:|---:|---:|---:|
| HR guide 5×5 | 28.36 dB | 0.8736 | 0.187 | 5.80 | 47% |
| residual b8 | 29.96 dB | 0.8704 | 17.204 | **171.31** | 61% |
| **kernel b16 r2** | **30.26 dB** | 0.8563 | **0.044** | **0.47** | **67%** |

Three things, in order of how much they change what to do next.

**The residual parameterisation was not the whole story, and the earlier
conclusion needs correcting.** On the old scenes the learned residual added
0.02 dB, and it was tempting to read that as the parameterisation being at
fault. Here the same shape adds 1.60 dB. Most of that null result was the data:
a residual over a filter has nothing to predict when the ground truth inside
every object is smooth and the G-buffer already segments it. Given content with
texture and shadow, there is a great deal to predict, and it predicts it.

**But it is unbounded, and it shows.** Its relative error is 17.2 against the
0.187 of the base it corrects — 92× worse than not running it at all, and
8× worse than the raw unfiltered input. The worst crop reaches 171. PSNR is an
absolute error in a compressed space and simply cannot see this: it reports the
same model as a 1.60 dB improvement. The failure is in the dark third of the
frame, where a residual added in compressed space and then decompressed turns a
small mistake into a large radiance, and where PSNR has almost no weight.

**The kernel model's worst case is better than the deterministic base's.** 0.47
against 5.80, and 364× better than the residual arm's. This is the property
claimed for the formulation rather than trained into it: the output is a convex
combination of samples the renderer measured, so there is no arithmetic by which
it can invent radiance that was not there. It is the difference between a
reconstruction that is usually right and one that cannot be very wrong.

The gap also widened as predicted, though not where expected. On PSNR the two
learned arms are close, 29.96 against 30.26. On relative error they are 393×
apart, and on detail retention it is 61% against 67% with the base at 47%.

## What this still does not settle

SSIM has gone inert on this content and should not be read as a result. Split by
how bright the crop is, the darkest third scores 0.9810 and the brightest 0.7905
— the hardest content getting the best mark, because C2 is an absolute constant
of 9e-4 in compressed space and a crop whose mean level is 0.039 has local
variance well below it, so the structure term divides two numbers that are both
approximately C2. The old scenes never exposed this: their darkest third
averages a level of 0.550. `metrics::ssim` is deliberately unchanged, since
every published figure is in terms of it, but the diagnostic now reports the
split so an inert number is visible as one.

Albedo demodulation is the obvious next experiment and now has a measurement
behind it. On these scenes, reconstructing in demodulated space and
re-modulating by the exact output-resolution albedo takes the deterministic
base from 47.5% to 65.4% detail retention — 18 points, for a technique that
measured as 0.1 points on the old data. The kernel path should gain from it too
and for the same reason: the high-frequency albedo is known exactly at output
resolution, so nothing needs to reconstruct it.

Training cost is worth noting. The kernel arm trains at 35.5 steps/s and the
residual arm at 4.7, on the same data and the same device, because the residual
target needs the 13×13 guided filter evaluated on the CPU for every crop of
every batch and the kernel target does not. Removing the deterministic filter
removed most of the training cost with it.

## The scenes, and why they had to change

The scenes cannot really discriminate these results. Blade's fallback
environment is a white 1×1 texture, so an open scene is lit by a uniform
furnace and nothing in it can be in shadow — none of the 4-spp validation
pixels fall below a displayed luminance of 0.10, and 87.8% sit in a single
quarter of the range. Every material is a constant colour, so albedo
demodulation, one of the larger wins in a production denoiser, measures here as
0.01 dB. That is a fact about the data.

`--canopy`, `--textures`, `--gloss` and `--ground-patches` address all of it,
off by default so existing figures still mean what they meant. On a 16-scene
probe with all four on, 32.6% of pixels fall below 0.10 where none did, and
they carry 22.8% of the displayed error against 5.6% of the error the project
reports — the blindness above, now measurable. The reference carries 2.3× the
gradient energy, and demodulation stops being free: 22 points of detail
retention where it was worth none.

Sealing the scene into a room was tried first and is wrong: the environment is
the only light the estimator importance-samples, so walling it off leaves three
small emissive spheres to be found by chance, and bilinear on the resulting
input scores 8.5 dB against 26.5 dB open. A canopy over part of the scene keeps
the sampling conditioned.

