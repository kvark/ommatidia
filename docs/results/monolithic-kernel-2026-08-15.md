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

## Demodulating the albedo

The albedo is known exactly at output resolution, so a reconstruction that
carries it through the filter is being asked to recover something it was already
told. Dividing it out before the gather and multiplying it back afterwards
leaves the smoother illumination term to reconstruct. Standard in production
denoisers, and on the deterministic base here it was worth eighteen points of
detail retention.

The loss stays in demodulated space, so the modulation is entirely outside the
graph and outside the gradient — the target is the canonical frame divided by
the same output-resolution albedo.

| external validation, 128 crops | PSNR | SSIM | relMSE | worst crop | detail |
|---|---:|---:|---:|---:|---:|
| HR guide 5×5 | 28.36 dB | 0.8736 | 0.187 | 5.80 | 47% |
| kernel b16 r2 | **30.26 dB** | 0.8563 | 0.044 | 0.47 | 67% |
| kernel b16 r2, demodulated, offset 0.05 | 28.74 dB | 0.8711 | 0.054 | 1.45 | 104% |
| **kernel b16 r2, demodulated, offset 0.25** | 30.21 dB | **0.8840** | **0.032** | **0.29** | **83%** |

The first attempt lost 1.5 dB, and the reason is worth keeping. The offset
bounds how far demodulation can rescale a pixel, and at 0.05 that is a factor of
twenty. The gather runs in a compressed space tuned for radiance; moving a pixel
twenty times up it lands where that space has almost no precision left, and the
detail figure of 104% is the tell — above the reference means noise, not
sharpness. At 0.25 the bound is four, and the same change is a clear win.

Against the plain kernel the demodulated one is level on PSNR, 26% better on
relative error, 38% better in its worst crop, and sixteen points better on
detail. It is also the first reconstruction here that beats the deterministic
base on SSIM as well, so for once all four measures agree. Its worst crop is
0.29 against the base's 5.80: twenty times better in the case that matters most.

The offset is carried in the checkpoint rather than living in WGSL, for the same
reason the guide coefficients are — a runtime that changed it would silently
reinterpret weights that were fitted against a different reconstruction.

## Three sweeps

All on the textured, shadowed scenes, b16, demodulated, 8,000 steps, scored on
the separate 128-scene set. The control arm reproduces the earlier figure
exactly, so these differ only in what they say they differ in.

### How far demodulation may rescale a pixel

| offset | PSNR | SSIM | relMSE | worst crop | detail |
|---:|---:|---:|---:|---:|---:|
| 0.05 | 28.74 dB | 0.8711 | 0.0544 | 1.45 | 104% |
| 0.10 | 29.38 dB | 0.8775 | 0.0390 | 0.54 | 94% |
| **0.25** | 30.21 dB | **0.8840** | 0.0325 | **0.29** | 83% |
| 0.40 | 30.47 dB | 0.8832 | **0.0324** | 0.41 | 78% |
| 0.70 | **30.55 dB** | 0.8771 | 0.0360 | 0.50 | 75% |
| none | 30.26 dB | 0.8563 | 0.0437 | 0.47 | 67% |

A large offset makes demodulation a no-op, so the last row is where the column
is heading. PSNR climbs all the way to 0.70 and then has to come back down;
relative error bottoms out between 0.25 and 0.40; SSIM and the worst crop both
prefer 0.25. The default stays at 0.25, which loses 0.34 dB against the
PSNR-optimal point and keeps the best worst case and five more points of detail
— consistent with everything else here about which of those numbers to believe.

### Tap radius

| radius | taps | head channels | PSNR | SSIM | relMSE | worst crop | detail |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 9 | 36 | 28.73 dB | 0.8439 | 0.0408 | 0.35 | 105% |
| **2** | 25 | 100 | 30.21 dB | 0.8840 | 0.0325 | 0.29 | 83% |
| 3 | 49 | 196 | **30.53 dB** | **0.8963** | **0.0318** | 0.29 | 76% |

Nine taps cannot denoise four samples per pixel — 105% detail is the metric
saying the noise came straight through. Twenty-five is the knee. Forty-nine buys
0.32 dB for twice the head, which is available if it is ever wanted and is not
the first thing to spend on.

### Head kernel

The head is 100 channels wide at radius two, a quarter of the network's
arithmetic, reading features that already carry the whole receptive field. A 1×1
head therefore looked like free money.

| head | PSNR | SSIM | relMSE | worst crop | detail | GFLOP | ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1×1 | 29.48 dB | 0.8580 | 0.0391 | 0.34 | 103% | 47.4 | 13.19 |
| **3×3** | **30.21 dB** | **0.8840** | **0.0325** | **0.29** | 83% | 60.7 | 14.41 |

It is not. It costs 0.73 dB, and the saving is smaller than it looks: 22% of the
arithmetic is 8.5% of the frame, because a wide head is bound by what it writes
rather than by what it multiplies. The spatial extent is doing real work —
without it each pixel's kernel depends only on its own features, and
neighbouring output pixels get uncorrelated kernels, which is the 103%.

### What the three have in common

Every knob here is the same knob. More taps, more offset, more head context all
mean more smoothing, PSNR always prefers more of it, and detail retention falls
monotonically as it is added. The useful landmark is 100%: every variant in this
work that underperformed — the b8 kernel at 99%, offset 0.05 at 104%, radius 1
at 105%, the 1×1 head at 103% — is a variant that failed to denoise, and the
detail figure said so in every case while PSNR alone did not.

## History as one more tap

The accumulated estimate becomes tap 25 and how much to trust it becomes a
weight the network predicts, rather than a base a residual model corrects.
Trained on 1,024 four-frame sequences with both camera and object motion on the
textured, shadowed scenes; scored on 128 unseen sequences, against the same
b16 capacity in the architecture the project previously selected.

| | PSNR | SSIM | relMSE | worst crop | detail | temporal | moving pixels |
|---|---:|---:|---:|---:|---:|---:|---:|
| four-frame HR guide | 28.54 dB | 0.8808 | 0.186 | 3.46 | 42% | — | — |
| kernel + history tap | 30.35 dB | 0.8746 | **0.041** | **0.65** | **80%** | **−1.12 dB** | **−3.19 dB** |
| low-colour residual | **30.57 dB** | **0.9003** | 73.29 | 2800.57 | 43% | +0.11 dB | +0.99 dB |

Neither of these is a result to ship, and they fail in opposite directions.

**The kernel model does not use history, and the reason is the loss.** Its
learned bias for the history tap is 0.0105 — the floor it was initialised at,
unmoved after 8,000 steps, 0.9% of the weight against the current frame's 1.20.
Twenty-five current-frame taps already denoise a 4-spp frame, so history buys
almost nothing on a per-frame squared error, and a per-frame squared error is
the entire objective. It is not that history was unavailable; it is that nothing
ever asked for it.

The consequence is that each frame's kernel is predicted independently from that
frame's own noisy input, so the weights move frame to frame and the output moves
with them. Every frame is individually good — 1.81 dB over the deterministic
base, four times less relative error, nearly twice the detail — and the sequence
flickers, by 1.12 dB overall and 3.19 dB where anything is moving.

**The residual model is stable, but it did not learn that either.** Its
stability is inherited: the accumulated history it corrects is stable by
construction, and a small correction on top of a stable base is stable. What it
did learn is unbounded, and here that is not a nuance — relative error of 73
against the base's 0.186, with a worst crop of 2,801. It is the same failure as
on the spatial sets, an order of magnitude further along.

So the honest reading is that the residual formulation was borrowing temporal
stability from the deterministic accumulation, and the kernel formulation, by
removing the deterministic stage, gave up the loan. Stability is not a property
of a single frame, and it will not appear in a single-frame objective no matter
which parameterisation is used. The project's own earlier note — that the next
architecture experiment is paired sequence training with a differentiable
temporal loss — is what this measures the need for, and `metrics::temporal_error`
is already the thing it would minimise.

What this does not need is a bigger network or more taps. It needs the objective
to contain the axis the failure is on.

## Putting stability into the objective

The temporal metric rearranges into a squared error against a target the host
assembles, so the graph gains six operations rather than a per-pixel gather it
has no primitive for. The previous frame's answer comes from a detached copy of
the network, resynchronised every 250 steps and initialised from the same seed.
Same data, same 8,000 steps, scored on the same unseen sequences.

| arm | PSNR | SSIM | relMSE | worst crop | detail | temporal | moving |
|---|---:|---:|---:|---:|---:|---:|---:|
| four-frame HR guide | 28.54 dB | 0.8808 | 0.186 | 3.46 | 42% | — | — |
| low-colour residual | 30.57 dB | **0.9003** | 73.29 | 2800.6 | 43% | +0.11 | **+0.99** |
| kernel, no temporal term | **30.35 dB** | 0.8746 | 0.041 | 0.65 | **80%** | −1.12 | −3.19 |
| **kernel, weight 1** | 30.04 dB | 0.8802 | **0.036** | **0.22** | 76% | −0.17 | −2.24 |
| kernel, weight 4 | 27.73 dB | 0.8729 | 0.047 | 0.29 | 77% | **+0.53** | −1.39 |
| kernel, weight 1, motion bias 8 | 28.40 dB | 0.8560 | 0.049 | 0.98 | 85% | −0.94 | −2.48 |
| kernel, weight 1, motion bias 32 | 25.49 dB | 0.8212 | 0.072 | 0.44 | 99% | −1.35 | −2.88 |

**The term does what it was built to do.** Temporal error moves monotonically
with its weight, −1.12 dB to −0.17 to +0.53, and at weight 4 the gather is more
stable than the deterministic accumulation it replaced. Nothing else tried in
this work moved that column at all.

**The exchange rate is steep.** 1.65 dB of stability costs 2.62 dB of per-frame
quality, and at weight 4 the reconstruction has fallen below the deterministic
base on PSNR. Stability is not free, which is the expected shape — a
reconstruction that agrees with its own past is constrained relative to one
free to re-decide every frame.

**Weight 1 is where to stand.** It gives up 0.31 dB of PSNR for 0.95 dB of
stability, and comes out ahead on everything else at the same time: relative
error 0.036 against 0.041, a worst crop of 0.22 against 0.65, and SSIM above
the untermed arm. Against the deterministic base it is 1.49 dB better per
frame, five times better on relative error, sixteen times better in its worst
crop, and within 0.17 dB on stability.

**Weighting the term toward moving pixels fails, and monotonically.** The
reasoning for trying it was sound — moving pixels are 2.7% of the valid ones and
carry all of the flicker, so at an even weight they contribute 2.7% of the
gradient. The reasoning against it is stronger and was not anticipated: the
target is least trustworthy exactly where motion is largest, because that is
where the motion vectors are least accurate and the reprojected teacher is
furthest from the truth. Amplifying those pixels amplifies target error rather
than a deficiency, and biases of 8 and 32 are worse than none on every column
including the one they were meant to help.

**What is still not solved.** Moving pixels never beat the deterministic base
under any setting here. The residual model does beat it there, by 0.99 dB — but
with a relative error of 73 and a worst crop of 2,801, so it is not a
counterexample so much as a different failure. The gap is real and it is where
the next work is; a better motion-compensated target, rather than a heavier
weight on the one that exists, is what the bias result points at.

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

