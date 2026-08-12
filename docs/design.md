# Ommatidia design

Ommatidia reconstructs a high resolution frame from a cheaply rendered low
resolution one. It is a portable DLSS replacement: the network runs through
[meganeura](https://github.com/kvark/meganeura) on Vulkan and Metal, so it has
no dependency on CUDA, on a vendor SDK, or on a specific GPU generation.

The first milestone is deliberately narrow: **spatial upscaling of a single
frame, no temporal context**. Everything below is written so that adding
history is an extension rather than a rewrite.

## Why diffusion, and what it costs

Upscaling is ill-posed. A 2x upscale has to invent three quarters of its
output pixels, and the honest answer is a distribution over plausible frames
rather than a single one. A regression network trained on L2 collapses that
distribution onto its mean, which is exactly the blur that makes naive neural
upscalers look worse than a good sharpening filter. Diffusion models the
distribution instead, which is why they set the quality bar on
super-resolution.

The cost is that sampling is iterative. A network that would fit a real-time
budget in one forward pass does not fit it in twenty. This is a real tension
and it is worth stating plainly up front rather than discovering it at
integration time:

- The **backbone** is a plain timestep-conditioned U-Net. Nothing about it is
  diffusion-specific except that one of its inputs is a noise level.
- The **objective** and the **sampler** are separable from the backbone. The
  same weights shape can be trained as an e-prediction diffusion model or, by
  fixing the timestep to zero and dropping the noise input, as a direct
  regressor.

So the plan was to establish the quality ceiling with diffusion first, then buy
the latency back through step distillation, with the direct regressor as the
always-available fast path and as the baseline that distillation has to beat.
`Objective` in `ommatidia::model` is the switch.

### That plan did not survive contact with the measurement

Matched backbone, matched data, matched everything but the objective, on 2400
scenes with 360 held out:

| objective | steps | held-out vs nearest |
|---|---|---|
| direct | 8000 | **+5.12 dB** |
| diffusion, 20 sampler steps | 12000 | +1.54 dB |
| diffusion, 1 sampler step | 12000 | +3.31 dB |

The raw logs are in `docs/results/`, and `scripts/curve.py` lines any two runs
up by step.

Diffusion was given half again as much training and lost by a wide margin. And
the sampler, the thing the whole formulation is built around, makes the result
*worse* the more it is used — monotonically, on the same checkpoint:

| sampler steps | 1 | 2 | 4 | 8 | 16 | 20 |
|---|---|---|---|---|---|---|
| dB | +3.31 | +3.31 | +2.49 | +2.72 | +1.99 | +1.55 |

With x0-prediction, a single DDIM step from the top of the chain returns the
model's `x0` estimate directly, with no re-noising. So the best thing this
diffusion model can do is to stop being a diffusion model.

The reason is that the premise at the top of this section is wrong for *this*
conditioning. Upscaling is ill-posed when the input is an image. It is much
closer to determined when the input is a low resolution render **plus the
renderer's own depth, normals, albedo, specular reflectance, and roughness** —
the network is not being asked to invent plausible detail, it is being asked to
resolve detail that the conditioning already implies. When the conditional
distribution is near a delta function, there is nothing for a sampler to
explore: iterating only accumulates the model's own error, and the capacity
spent learning to denoise at every noise level is capacity not spent on the one
mapping that matters.

Caveats, because this is one experiment: a single seed, one dataset, one model
size, one scale factor, and a scene distribution that is procedural rather than
authored. A harder distribution — thin geometry, strong specular detail, a
larger scale factor — would push back toward ill-posed, and the conclusion
could change with it. What is not in doubt is that on this problem, as posed,
the diffusion machinery cost quality rather than buying it.

**So the direct objective is the main line**, not the fallback, and the
sub-pixel residual formulation below stands on its own without the noise
schedule. The diffusion path stays because the backbone is shared and the
comparison is worth being able to re-run, and because the caveats above are
real. It is no longer the thing to beat.

## Formulation: sub-pixel residual diffusion

The naive setup runs the U-Net at output resolution. For a 4K target that is
four times the work of running it at 1080p, which is the wrong place to spend
the budget: the conditioning signal only exists at low resolution anyway.

Instead, ommatidia diffuses in a **sub-pixel space** at input resolution.

Let `S` be the scale factor, `(W, H)` the low resolution extent. The target
high resolution image `Y` of shape `[3, S*H, S*W]` is rearranged into
`[3*S^2, H, W]` by space-to-depth: sub-pixel `(dy, dx)` of low resolution pixel
`(y, x)` becomes channel `c*S^2 + dy*S + dx`. This is a pure reindexing, no
interpolation and no information lost.

The network then predicts a **residual** over nearest-neighbour upsampling:

```
target[c, dy, dx, y, x] = Y[c, S*y + dy, S*x + dx] - LR[c, y, x]
```

Nearest is chosen over bilinear as the base because it is exactly reproducible
between the trainer and the shader that reassembles the output, with no edge
convention to get wrong.

Three things fall out of this:

- The entire network runs at low resolution. Input, every level, and output.
- Diffusion is well posed: `x_t` and the predicted noise have the same shape.
- The output head is a free reindex. The unpack shader writes sub-pixel
  channel `c*S^2 + dy*S + dx` to high resolution texel `(S*x + dx, S*y + dy)`,
  adding the low resolution pixel back as it goes.

It also puts the diffusion where the uncertainty actually is. The low frequency
content is already determined by the input; only the sub-pixel detail is being
invented, so that is the only thing the noise schedule has to cover.

### Two things this formulation gets wrong if you are not careful

Both were found by measuring reconstruction quality rather than training loss,
and both produced a *falling* loss with a sampler that returned pure noise.
They are recorded here because the failure gives no hint of the cause.

**The residual is not unit scale.** Most of a frame is already correct at low
resolution, so the residual's standard deviation is a few hundredths — measured
at 0.057 on the first dataset. A diffusion schedule assumes unit variance data,
and against unit noise a signal that small is invisible at nearly every
timestep. The network settles on the degenerate solution and the sampler
returns noise. The fix is a gain that brings the residual to unit variance,
measured from the training set by `batch::estimate_gain` and carried in the
checkpoint so inference divides by exactly what training multiplied by. This is
the same correction latent diffusion models apply to their latents.

**e-prediction cannot be sampled here.** Recovering the clean signal from a
predicted noise means dividing by `sqrt(alpha_bar)`, which at the end of a
cosine schedule is around `1e-3`. That multiplies the network's error by a
thousand, at exactly the first sampling step, which is where the network knows
least. Predicting `x0` instead never performs that division — the corresponding
recovery divides by `sqrt(1 - alpha_bar)`, which approaches 1 where the other
approaches 0. Switching the parameterization moved reconstruction error from
0.29 to 0.001 with no change to training. It also unifies the objectives:
`Objective::Direct` is x0-prediction with the noise level pinned at zero.

## Conditioning: use the G-buffer

This is the main structural advantage a renderer has over photographic
super-resolution, and the reason a neural upscaler for rendering can beat a
generic one. A renderer is not handed an image, it is asked to produce one, and
it knows things about the frame that are not recoverable from pixels:

- **Depth and normals** give exact geometric edges. An upscaler does not have
  to guess where a silhouette is, it is told, at sub-pixel accuracy.
- **Diffuse albedo and specular F0** separate material from lighting. Texture
  detail that survives at low resolution comes back through albedo rather than
  having to be hallucinated.
- **Roughness** predicts how sharp a specular highlight should be, which is the
  single hardest thing for a spatial upscaler to get right.

All of these are cheap. Producing them at low resolution is free, they are
already in the G-buffer.

Blade hands them over through `RayTracer::view_gbuffer`, and the generator's
probe reads them straight into the planar layout a record uses. The shading
normal comes from the `basis` quaternion rather than the flat normal, so
normal-mapped detail survives; a ray that hit nothing is recorded as a very
large depth and a zero normal, which is not a direction any surface can have
and so marks the sky unambiguously.

Only the input side carries a G-buffer. The reference is what the network has
to reach, and it is reached in colour.

### It measurably helps

Two runs over the same 2400-scene set, same seed, same crops, same batch order,
differing only in which channels reach the network — `--color-only` is the
other arm, so no dataset is regenerated and nothing else can drift. Scored on
360 held-out scenes at every thousand steps:

| step | colour + G-buffer | colour alone | difference |
|---|---|---|---|
| 938 | +3.43 dB | +3.19 dB | +0.24 |
| 1876 | +4.41 | +4.00 | +0.41 |
| 2948 | +4.79 | +4.29 | +0.50 |
| 3886 | +4.99 | +4.48 | +0.51 |
| 4958 | +5.07 | +4.56 | +0.51 |
| 8000 | **+5.12** | **+4.61** | **+0.51** |

Half a decibel, holding steady across the whole run, for channels the renderer
had already produced. An earlier version of this comparison on a 192-scene set
measured 0.66 dB; the gap narrowing slightly as the data grows is what one
would expect, since more scenes give the colour-only arm more chance to learn
what the G-buffer would otherwise have told it.

Absolute numbers are not comparable between the two sets — the nearest baseline
itself moved from 0.0033 to 0.0041 when boxes entered the scene distribution,
because straight silhouettes carry high frequency content that spheres do not.
Only the within-set difference means anything.

The first attempt at this comparison scored one crop of one *training* sample
and reported the two arms as indistinguishable. Both halves of that were wrong:
the score was in-sample, and one 64x64 tile is far too small and too lucky to
separate anything — the tile it happened to pick was a flat wall, where the
nearest baseline scores 0.0009 against the 0.0033 it scores across the held-out
set. Hence `Split`, and hence scoring over a grid of crops. It is worth being
suspicious of any number produced before that was in place.

### What is not done yet

The reference render also fills a G-buffer, at output resolution, and that is
the more interesting half. A depth and normal prepass at full resolution costs
far less than shading at full resolution, so a deployed renderer could supply
high resolution geometry for nearly nothing — and the sub-pixel formulation has
a natural place to put it, since space-to-depth turns a high resolution plane
into `scale^2` channels at input resolution, exactly the shape the network
already consumes. That is the next thing to try after the input-side G-buffer
is shown to earn its keep.

## Data generation

Training data comes from [blade](https://github.com/kvark/blade), which has
both halves of the pair already:

- `RenderMode::RealTime` is the raw ReSTIR estimator, one sample per pixel,
  with Blade's SVGF pass disabled. This is the input: Ommatidium replaces the
  built-in denoiser rather than learning to upscale its output.
- `RenderMode::Canonical` is `RayTracer::path_trace`: full paths, BSDF sampling
  with next event estimation, MIS, accumulated over many frames with no reuse
  and no denoising. This is the ground truth.

The `.omd` header records which of those renderer paths produced the input.
Version-1 files are identified as SVGF because they predate raw capture, and
the trainer rejects them by default. This turns the most expensive possible
configuration mistake — fitting a supposed replacement to the filter it is
meant to replace — into an immediate error.

Both are driven headless. For each sample the generator builds a fresh
procedural scene, picks a camera pose, renders the low resolution input, then
renders the high resolution reference by accumulating canonical samples until
the configured count is reached, and writes one record. Scenes are randomised
per sample rather than viewed from many angles, so the network sees layout
variety rather than one scene memorised.

Capturing this needs the renderer to hand back radiance rather than a picture,
which is what `PostProcConfig::tone_map` is for: cleared, the post process
leaves the composed linear radiance alone and skips the display transfer
function, and an `Rgba32Float` target holds it unclamped. The generator reports
the peak radiance it saw for exactly this reason — a peak pinned at 1.0 means
something clamped and the dataset is quietly worthless.

Ground truth being an unbiased path trace rather than a supersampled raster is
worth more than it might look. The network is not being taught to imitate a
sharper version of the same estimator, it is being taught what the estimator is
converging to, which means it can learn to remove the estimator's bias and not
just its aliasing.

The corollary is that a change to the canonical renderer invalidates the
dataset. Blade's `732d0ef` fixed next event estimation losing the share of the
contribution it had held back for a BSDF sample that a terminating path never
takes, which made every reference frame slightly dark. A set generated before
it teaches the network to reproduce that bias. Regenerate rather than reuse.

Scenes carry spheres and boxes over a ground plane, lit by an ambient
environment and a few emissive spheres, with material, layout, and viewpoint
randomised per sample. The boxes matter more than the count suggests: spheres
never present a straight silhouette at an arbitrary angle, which is exactly
where a spatial upscaler staircases, nor a hard normal discontinuity. The
ground's tone and roughness vary per scene too, since a floor of one fixed
brightness in every sample is something the network can learn instead of the
geometry.

## File format

`.omd`, described in `ommatidia::dataset`. A fixed 64 byte header followed by
tightly packed records. Everything is `f16`, planar, channel-major, which is
already NCHW so the trainer can hand a batch to meganeura without shuffling.

The header names which planes are present, so a dataset generated with more
channels than a given model consumes stays readable, and the trainer errors
loudly rather than silently misinterpreting a plane if they disagree.

What is stored is what the renderer produced — linear radiance, view-space
distance, unit normals — and *not* anything preconditioned for the network.
That distinction is worth being deliberate about, because getting it backwards
is an easy and expensive mistake.

The network does want bounded inputs, so radiance is range-compressed by
`x / (1 + x)` and depth is inverted to `1 / (1 + d)`. The temptation is to
apply those on write and store the result. Doing so would destroy the data:
`f16` spends its bits on an exponent and so holds radiance at roughly 0.1%
relative precision across its entire range, which is exactly what high dynamic
range needs, whereas compressing first crushes every bright value up against
1.0 where `f16` steps by 1/2048. A radiance of 1000 and one of 2000 would land
on adjacent representable values.

So the transforms live in `ommatidia::transform`, applied on load, and mirrored
by the pack shader so the trainer and the runtime agree exactly. `f16`'s only
real limit, saturation at 65504, is left as a clamp on write.

## Runtime

Blade users hand over their `Arc<blade_graphics::Context>`. Meganeura's
`SessionConfig::gpu` takes it directly, so the network executes on the host's
own device and queue with no second context, no external memory import, and no
cross-device copy. This requires that both resolve to the same `blade-graphics`
crate, which the workspace `[patch]` section enforces.

Per frame:

1. **Pack.** One compute dispatch reads the host's colour and G-buffer texture
   views and writes an interleaved-to-planar `f32` tensor into the session's
   input buffer, applying the range compression above. Format conversion,
   normalisation, and layout change all happen here, so the host is free to
   hand over whatever texture formats it already has.
2. **Step.** `Session::step()`, once per sampler step.
3. **Unpack.** One compute dispatch scatters the sub-pixel output to the high
   resolution target, adding the nearest-neighbour base and undoing the range
   compression.

Pack and unpack are ommatidia's own WGSL, dispatched onto the caller's command
encoder, so the whole thing is one recorded sequence with no CPU roundtrip.

## Training

`scripts/curriculum.sh` drives a long run. Two things it does are worth
repeating anywhere else this gets run.

It **serialises** the runs. Two trainings on one GPU contend for the same cores
and the same memory, so running them one after another costs nothing in
throughput and keeps the footprint to one model — which matters, because this
device is often shared with something else entirely.

It **calibrates before sizing**. Step rate here is set by contention, not by
model size: the same network measured 8.3 steps/s on an idle device and 1.1
steps/s beside another training process. Sizing a run from a figure measured in
the other regime is wrong by an order of magnitude, and the cosine learning rate
schedule needs the total step count up front, so it cannot be corrected
halfway.

The trainer scores a held-out split periodically rather than only at the end,
which is what makes a multi-hour run steerable, and checkpoints on the same
cadence so a crash costs one interval rather than everything.

## The latency problem

The premise is a real-time budget, and the network is nowhere near one. All
figures below are measured at 720x720 input, which is 1440x1440 out — the same
2.07 million pixels as a 1080p frame — on an otherwise idle 7900 XT.

| shape | params | GFLOP | ms @1080p, start | ms now | held-out dB |
|---|---|---|---|---|---|
| base 64, 3 levels, 2 blocks | 6.50M | 1096 | 656 | 122 | +5.08 |
| base 32, 3 levels, 1 block | 1.15M | 182 | 131 | 35 | — |
| **base 24, 3 levels, 1 block** | **649k** | **104** | **89** | **28** | **+5.04** |
| base 16, 2 levels, 1 block | 72k | 33 | 42 | **15** | +4.30 |
| base 8, 2 levels, 1 block | 19k | 9 | 21 | 8.8 | — |

Quality is on 128 held-out crops, every shape trained to 20000 steps except the
reference, which had 8000 and had plateaued. **base 24 matches the reference
within noise on a tenth of the arithmetic and a quarter of the frame time**, so
the whole middle of this table was a measurement artefact of the earlier sweep,
which gave every shape 5000 steps and so compared undertrained large networks
against nearly-converged small ones. It read +4.10 for base 24 and +2.92 for
base 16; trained out they are +5.04 and +4.30.

Taken together with the kernel work, the frame went from 656 ms to 28 at equal
quality — 23x — of which 5.4x was the kernels and the rest was not needing the
larger network in the first place.

The two kernel fixes below account for 5.4x of that on the reference shape and
2.3x at the small end, with the weights untouched: a checkpoint trained before
the changes scores 0.001247 where it scored 0.001248, which is float
reassociation.

28 ms is still 14x a two millisecond budget, and utilisation runs from 1.7% to
15%, so there is a lot left. But the shape of the problem has changed: it is no
longer obvious that quality has to be traded for it.

Where the time goes, from `gpu_profile`'s per-pass timings:

| | share |
|---|---|
| convolution (Winograd transforms, batched matmuls, GEMM) | ~62% |
| GroupNorm + SiLU | ~34% |
| everything else | ~4% |

### Two kernels were leaving the device idle

Both were shaped for training, where a large batch supplies the parallelism,
and both starve at a batch of one.

**GroupNorm was parallel in the batch, not in the image.** It launched
`batch * num_groups` workgroups of 256 threads — at inference with eight
groups, 2048 threads, whatever the resolution, on a tensor of millions of
elements. Raising the group count showed the shape of it directly, since that
changes the parallelism and nothing else: 340 ms at 8 groups, 244 at 32, 230 at
64. That is not a usable fix, because groups are a modelling choice and
training base 24 with one channel per group cost 1.55 dB. The fix is to split
each group's elements into slices with a workgroup each, in two passes — one
writing partial sums, one combining them and normalising. The frame went from
340 ms to 239.

**The Winograd transforms read and wrote a megabyte apart per lane.** With
GroupNorm out of the way the input transform stood at 68% of the frame, taking
eight times the batched matmul it exists to make cheaper. Both transforms
indexed threads as `tile_idx = idx / channels`, putting neighbouring threads on
neighbouring channels, which are `H * W` apart in the input and `total_tiles`
apart in the transform domain. Every wave scattered, on the load and the store
alike. Swapping the decomposition so neighbouring threads take neighbouring
tiles makes the store contiguous and the load walk a row: 239 ms to 59.

The second one only became visible once the first was fixed, and the first only
became visible once the profile was read at all. Worth remembering before
concluding that a stack is simply slow.

### What is left, and what will not help

After both fixes the profile at 512x512 is 60 ms, and convolution is 82% of it
— which is where the arithmetic is, so that is the right shape. GroupNorm is
down to 11% from 34%, and the pointwise operations are 7%.

The largest single item is the Winograd batched matmul, 41% across the three
widths. It is **memory bound, and cooperative matrix would not help it**: at
level 0 it moves 537 MB to do 8.6 GFLOP, an arithmetic intensity of 16 FLOP per
byte against a ridge point of 64 on this device. It achieves 405 GB/s of a
possible 960, so there is perhaps 2x of tuning in it, but no more.

The reason is inherent to the algorithm. Winograd F(2,3) carries sixteen values
for every four outputs, so the transform domain is **four times** the size of
the activations. It trades arithmetic for bandwidth, which is the resource that
is actually scarce here. It still wins — 59 ms against 84 with it disabled —
but it wins less than it would on a compute-bound workload.

So the two levers left both move less data rather than doing less arithmetic:

- **`f16` activations.** Halves the traffic everywhere, and the profile is
  bandwidth bound almost end to end. Three of meganeura's seventy-seven shaders
  currently mention `f16`, and none of them are in the convolution path, so this
  is a real project rather than a flag.
- **Implicit-GEMM Winograd**, folding the transforms into the matmul so the
  transform-domain tensors are never written to memory at all. That removes the
  4x expansion from the bandwidth bill entirely, and is the larger rewrite.

Neither is a small change, and the profile should be re-read after either,
because both of the fixes above only became visible once the one before it was
out of the way.

### Barriers are not the bottleneck, yet

Meganeura groups dispatches by dependency level and puts a global barrier
between groups. The network is a chain, so this comes to 145 dispatches in 117
groups — 1.24 dispatches per group, which is close to one barrier each.

That sounds bad and currently is not: forcing one dispatch per group with
`MEGANEURA_SERIAL_DISPATCH`, which adds 28 more barriers, changes the frame
time by less than the measurement noise (340.6 and 340.7 ms against 340.4 and
341.6). The chain's dependencies are real, so a finer-grained barrier would
have little to overlap.

It becomes a problem at the target. A hundred-odd global barriers at even ten
microseconds apiece is most of a 2 ms budget, so reaching real time means
fewer dispatches — fusing convolution with the normalisation and activation
around it — rather than cheaper barriers.

### A correction

An earlier version of this section reported 175 ms at 1080p and 10% of peak.
Both were wrong: the benchmark inherited the small configuration the smoke
tests use, so it was costing a base-16 two-level network with four
conditioning channels and calling it the trained one. The real figure is 656
ms and 2.7%. The benchmark now builds the shape explicitly and measures at the
1080p pixel count instead of extrapolating from a quarter of it.

A second correction: a 9x figure for what shrinking the architecture buys was
measured while another job had the GPU. Idle, it is 32x.

## Roadmap

**Now.** Static frame, no history. Everything above.

**A network that can actually run.** Ahead of everything below, for the reason
above: quality work on something that takes 175 ms per frame is quality work on
something nobody can ship. Fewer channels at full resolution is the first
lever, since that is where both the arithmetic and the bandwidth are.

**Temporal context.** The largest quality win available, and the reason DLSS
works as well as it does: motion vectors plus a history buffer turn upscaling
from invention into accumulation. Blade already writes a motion vector target,
and the format reserves the plane. The generator gets more involved because
samples stop being independent, so it has to emit camera trajectories rather
than isolated poses, and the record has to carry the previous frame's output.

**Step distillation.** Deprioritised. It was going to buy the diffusion path's
latency back, but the measurement above says one step already beats twenty on
this problem, so there is nothing to distil — the fast path and the good path
turned out to be the same path.

**Standalone.** Drop the `blade-graphics` requirement from the public API and
connect at raw Vulkan: the host passes `VkImage` handles and a `VkCommandBuffer`
to record into. Blade already imports external memory, so the internals stay
the same and it is the surface that changes. A C ABI over that surface is what
makes C++ engines callable.
