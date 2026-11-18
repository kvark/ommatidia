# One recurrent reconstruction model (v4)

The only maintained model is the F1 replacement: a biased, four-level U-Net,
width 16, four latent channels, 2× output scale. It predicts six direct radiance
channels, two 25-tap kernels, two kernel/direct mix weights, two history gates
and four latent channels per output pixel. Config v3 and its weights are
rejected; the last v3 source is tagged locally `archive/v3-guide-residual`,
and its [built control runtime](archive/README.md) remains usable offline.

Each frame supplies LR diffuse illumination/specular radiance, LR normal/depth,
projection jitter, and HR normal/depth, diffuse albedo, F0, roughness and motion.
Motion features are zero on hard resets, when there is no previous reconstruction;
warm-frame vectors and the geometric warp retain their observed pixel units.
HR signals are subpixel-packed onto the LR grid. Sample offsets describe each
output pixel relative to its jittered input sample. Radiance is encoded as
`c(exposure * radiance)`, where `c(x) = x/(1+x)`; nonnegative depth uses `c(depth)`.
The host supplies positive per-frame exposure, which never changes output units.
Targets have a separate API and cannot enter observation packing. F1 additionally
packs linear `exposure * radiance` over a row-major 5×5 LR neighborhood, once
per LR pixel and RGB lobe channel. Frame/crop edges use nearest-edge extension;
this is coordinate addressing, not a radiance clamp. The graph reuses these
observations for each HR subpixel. Compressed features are never inverted.

Previous linear lobes, latent and normal/depth are bilinearly warped **inside
the graph**. The GPU constructs taps; out-of-frame taps are dropped and the rest
renormalized. No taps, or a camera cut, means invalid history. There are no fixed
filters, ages, moments, material/depth rejection thresholds or reactive weights.

```text
features = U-Net(observations, c(exposure * warped_lobes),
                 warped_latent, warped_normal_depth, valid)
z, k, m, a, s = five biased 1×1 heads(features)
direct   = exp(clamp(z, -16, 11))
kernel   = sum(softmax(k) * exposure_normalized_LR_neighborhood)
spatial  = ((1 - sigmoid(m)) * direct + sigmoid(m) * kernel) / exposure
alpha    = sigmoid(a) * valid
lobes    = alpha * warped_lobes + (1 - alpha) * spatial
latent   = tanh(s)
RGB      = albedo * diffuse + specular + emission
next     = lobes, latent, current_normal_depth, current_albedo
```

The radiance bias is the log of each training-corpus mean normalized lobe,
restricted to the decoder's representable log range. Gate bias is logit(0.8).
Head weights, kernel/mix logits and latent bias start at zero: uniform initial
kernels and an equal learned mix, with no fixed filter or blend in inference.
Kernel and mix weights are independent per lobe/HR pixel, shared across that
lobe's RGB. Softmax is over 25 taps, not spatial pixels or colors. Encoder
kernels use Kaiming initialization and zero biases. No old checkpoint initializes
F1. Old direct-only v4 checkpoints fail the exact parameter-layout check;
training bundle schema 4 also rejects pre-F1 optimizer/state resumes.

Four-frame tied-weight BPTT differentiates through both lobes and latent.
The objective retains compressed RGB and lobe MSE, exposure-normalized linear
RGB, coarse linear structure and valid-history temporal change error. Outputs
are `[lobes, latent, state]` in inference and `[loss, lobes, latent, state]` in
training; optional final alpha is diagnostic only. Meganeura differentiates the
first, scalar training output while preserving carried outputs.
Training uses eight persistent 64² LR crops and four-frame windows, excluding a
four-pixel HR loss margin at artificial crop edges (real image edges stay supervised).
Observations/maps are packed on GPU; detached state
is carried with GPU copies. Mean-gradient accumulation takes one clipped Adam
step per batch. Evaluation shares parameter storage. Read-only memory-mapped
captures feed a bounded crop-prefetch worker; radiance gains stay fixed per
cursor life. Checkpoints include Adam, schedule, cursor/RNG state and capture
hashes; weights-only warm starts are rejected. Phase 3 passes the throughput and
5,000-step full-corpus gates, but its checkpoint remains well below v3 quality.

At 128×128 → 256×256: **132 encoder input channels, 657,792 parameters,
1,076,887,552 convolution MACs = 2.15378 GFLOP/frame = 32,864 FLOPs/output pixel**.
These are dense forward-convolution counts, not timing measurements; they omit
activations, kernel weighting/reduction, warps and backward. The extra neighborhood
input is separate from the encoder. This exceeds archived v3's 2.08876 GFLOP/frame;
performance remains reported, not gated. The fixed-budget F1 development
scores remain unverified.

The renderer records `Native::record_prepare(..., jitter, exposure)`,
`native.session.record(&mut encoder)`, then `record_resolve` into its own started
Blade encoder (`manual_barriers: false`) on the shared context. Renderer passes
can produce observations before them and consume reconstructed RGBA afterward,
without intermediate submissions or CPU waits. Submit once and call
`native.session.track_submission(sync)` so later host access and destruction
wait for the caller's GPU work. Keep the renderer's normal frame fences before
reusing command buffers or host-visible uploads; tracking does not wait by itself.
Submit recorded frames in order: resolve advances history bookkeeping when
recorded, so do not discard recorded frames. See the
[runnable example](../ommatidia/examples/blade.rs). Only packing and RGB/state
resolve remain in WGSL. Reset on cuts; recreate on extent changes.
`process` is an offline upload/readback convenience, not a latency benchmark.
Inference uses f32 operands and unfused dispatches. Blade's pinned Naga fork
includes the workgroup layout correction.
Tests independently check CPU/WGSL packing and taps, 24-frame recurrent HDR/reset
parity, exposure equivariance, f64 loss/every-parameter gradients under both
lowerings, production-size directional gradients and bit-exact reload.
Current evidence and outstanding gates are in [the run ledger](experiments.md).

## Architecture review, November 18, 2026

Keep F1 as the single quality experiment. A kernel can average observed light
even on a cut. The direct head can estimate radiance outside the range of the
25 local observations. Reprojection lets both lobes reuse prior samples
when the same surface remains visible. Primary-surface motion is a weaker cue
for reflected highlights and lighting changes. Inspect the separate specular
outputs and the lighting-final frames when judging whether recurrence helps.

The predecessor shows why this distinction matters. On the same development
sequences, direct-only 16×4 seed 2 reduced causal temporal MSE to 0.81× v3
while warm smooth-crop MSE rose to 3.25× (95% interval 2.44–3.71) and FLIP
rose from 0.108 to 0.171. A stable, blurry or biased image can score well on
the first measure. The F1 decision still needs the fixed 60k checkpoint, both
reset protocols and paired comparisons to v3. [Decision A](phase5-results.md#decision-a-owner-handoff).

The 200 synthetic scenes cover motion, cuts and known targets well enough for
this development decision. Ten development sequences, with cold smooth crops
from four scenes, give limited evidence about generalization. A game-quality
claim needs held-out, game-like sequences with glossy motion, disocclusion,
thin geometry and changing light, paired with converged references. Replay
identical sparse inputs through each reconstructor and disclose differences in
guides, path depth, history resets and exposure. [Corpus](training-corpus.md)
and [evaluation protocol](evaluation.md).

Two parts of the graph expand the data stored for every pixel.

- Reprojection writes four u32 indices and four f32 weights for **each of 17
  state channels**. That costs **544 bytes per output pixel** in input buffers.
  All channels share the same geometry. A warp that computes the channel offset
  during sampling could consume one set of taps per pixel.
- The filter expands six radiance channels across 25 taps for every HR pixel.
  One expanded f32 tensor represents **600 bytes per output pixel** (1.24 GB
  for nominal 1080p). This is a logical tensor size. Peak allocation also
  depends on lowering and buffer reuse. A fused stencil could load the LR
  neighborhood directly and reduce into six output values.

The current release build requested **291 MB** of resident buffers at just
128×128 → 256×256, including the offline upload and readback buffers. On the
Ryzen 5 9600X integrated Radeon (RADV, Mesa 26.2.3), the summed GPU passes
took **82.25 ms p50 and 85.09 ms p95**. Median stage times were 14.52 ms
for preparation, 66.49 ms for the network and 1.29 ms for resolve. The RX
7900 XT was busy with another workload. These timestamps exclude CPU work,
compilation and gaps between submissions. The benchmark used the full F1
shape with two-update smoke weights, two warmup sequences and five measured
replays of the same 16 frames. [Timing and memory report](f1-timing.json).

Both optimizations can preserve the learned weights and equations. Qualify
each against F1 forward outputs, gradients and recurrent rollouts. Measure
allocation and GPU p50/p95 on the same clip and adapter. A shipping comparison
also needs GPU timestamps around the caller's single-encoder reconstruction,
at a common output resolution, and a matched ray-reconstruction baseline.

Apex publishes frame-generation timings on the RTX 5090 at 4K. Its current
2X table gives 1.01 ms per synthesized frame for ANR FG and 1.26 ms for DLSS
FG. Its ray-reconstruction and super-resolution betas are listed as upcoming.
The transferable idea is to replay identical inputs and compare synchronized
sequences across devices. [Apex measurements](https://apexcompute.com/neural-rendering)
and [ANR release and replay tooling](https://github.com/apex-compute/ANR).

REGEN trains a lightweight game-image translator on paired teacher outputs.
Our paired target is already available from the high-sample renderer.
OpenDLSS-NR implements same-resolution generative rendering and checks
intermediate activations against fixtures. Its per-layer parity checks would
help qualify fused or reduced-precision execution of F1. [REGEN](https://github.com/stefanos50/REGEN)
and [OpenDLSS-NR](https://github.com/maanHimself/OpenDLSS-NR).
