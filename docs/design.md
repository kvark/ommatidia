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

So the plan is to establish the quality ceiling with diffusion first, then buy
the latency back through step distillation (consistency / adversarial
distillation to 1-4 steps), with the direct regressor as the always-available
fast path and as the baseline that distillation has to beat. `Objective` in
`ommatidia::model` is the switch.

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

Two runs over the same 192-sample set, same seed, same crops, same batch order,
differing only in which channels reach the network — `--color-only` is the
other arm, so no dataset is regenerated and nothing else can drift:

| conditioning | training loss | held-out MSE | vs nearest |
|---|---|---|---|
| colour + G-buffer | 0.163 | 0.000718 | 6.64 dB |
| colour alone | 0.199 | 0.000836 | 5.98 dB |

0.66 dB, a 14% reduction in reconstruction error, for channels the renderer
had already produced.

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

- `RenderMode::RealTime` is the ReSTIR estimator with a denoiser, one sample
  per pixel. This is the input, and critically it is the *actual* renderer the
  upscaler will be deployed behind, noise characteristics and all.
- `RenderMode::Canonical` is `RayTracer::path_trace`: full paths, BSDF sampling
  with next event estimation, MIS, accumulated over many frames with no reuse
  and no denoising. This is the ground truth.

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

## Roadmap

**Now.** Static frame, no history. Everything above.

**Temporal context.** The largest quality win available, and the reason DLSS
works as well as it does: motion vectors plus a history buffer turn upscaling
from invention into accumulation. Blade already writes a motion vector target,
and the format reserves the plane. The generator gets more involved because
samples stop being independent, so it has to emit camera trajectories rather
than isolated poses, and the record has to carry the previous frame's output.

**Step distillation.** Bring the sampler down to 1-4 steps, measured against
the direct regressor baseline.

**Standalone.** Drop the `blade-graphics` requirement from the public API and
connect at raw Vulkan: the host passes `VkImage` handles and a `VkCommandBuffer`
to record into. Blade already imports external memory, so the internals stay
the same and it is the surface that changes. A C ABI over that surface is what
makes C++ engines callable.
