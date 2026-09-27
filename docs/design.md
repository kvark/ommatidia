# One recurrent reconstruction model (v4)

The only maintained model is a biased, three-level U-Net: width 16, four latent
channels, 2× output scale. It predicts six spatial radiance channels, two history
gates and four latent channels per output pixel. Config v3 and its weights are
rejected; the last v3 source is tagged locally `archive/v3-guide-residual`,
and its [built control runtime](archive/README.md) remains usable offline.

Each frame supplies LR diffuse illumination/specular radiance, LR normal/depth,
projection jitter, and HR normal/depth, diffuse albedo, F0, roughness and motion.
HR signals are subpixel-packed onto the LR grid. Sample offsets describe each
output pixel relative to its jittered input sample. Radiance is encoded as
`c(exposure * radiance)`, where `c(x) = x/(1+x)`; nonnegative depth uses `c(depth)`.
The host supplies positive per-frame exposure, which never changes output units.
Targets have a separate API and cannot enter observation packing.

Previous linear lobes, latent and normal/depth are bilinearly warped **inside
the graph**. The GPU constructs taps; out-of-frame taps are dropped and the rest
renormalized. No taps, or a camera cut, means invalid history. There are no fixed
filters, ages, moments, material/depth rejection thresholds or reactive weights.

```text
features = U-Net(observations, c(exposure * warped_lobes),
                 warped_latent, warped_normal_depth, valid)
z, a, s  = three biased 1×1 heads(features)
spatial  = exp(clamp(z, -16, 11)) / exposure
alpha    = sigmoid(a) * valid
lobes    = alpha * warped_lobes + (1 - alpha) * spatial
latent   = tanh(s)
RGB      = albedo * diffuse + specular + emission
next     = lobes, latent, current_normal_depth, current_albedo
```

The radiance bias is the log of each training-corpus mean normalized lobe,
restricted to the decoder's representable log range. Gate bias is logit(0.8).
Head weights and latent bias start at zero; encoder kernels use Kaiming
initialization and zero biases. No v3 checkpoint initializes v4.

Four-frame tied-weight BPTT differentiates through both lobes and latent.
The objective retains compressed RGB and lobe MSE, exposure-normalized linear
RGB, coarse linear structure and valid-history temporal change error. Outputs
are `[lobes, latent, state]` in inference and `[loss, lobes, latent, state]` in
training; optional final alpha is diagnostic only. Meganeura differentiates the
first, scalar training output while preserving carried outputs.
Training uses eight persistent 64² LR crops and four-frame windows, excluding a
four-pixel HR loss margin. Observations/maps are packed on GPU; detached state
is carried with GPU copies. Mean-gradient accumulation takes one clipped Adam
step per batch. Evaluation shares parameter storage. Read-only memory-mapped
captures feed a bounded crop-prefetch worker; radiance gains stay fixed per
cursor life. Checkpoints include Adam, schedule, cursor/RNG state and capture
hashes; weights-only warm starts are rejected. Phase 3 throughput and full-corpus
training gates are still pending, not established by correctness tests.

At 128×128 → 256×256: **132 input channels, 174,576 parameters, 814,743,552
convolution MACs = 1.62949 GFLOP/frame = 24,864 FLOPs/output pixel**.
These are dense forward-convolution counts, not timing measurements; they omit
activations, warps and backward. The archived v3 count was 2.08876 GFLOP/frame.

The renderer submits `Native::record_prepare(..., jitter, exposure)`, the shared
Meganeura inference session, then `record_resolve`, in queue order. Only packing
and RGB/state resolve remain in WGSL. Reset on cuts; recreate on extent changes.
`process` is an offline upload/readback convenience, not a latency benchmark.
Inference keeps the verified f32/unfused policy and pinned Naga layout fix.
Tests independently check CPU/WGSL packing and taps, 24-frame recurrent HDR/reset
parity, exposure equivariance, f64 loss/every-parameter gradients under both
lowerings, production-size directional gradients and bit-exact reload.
Current evidence and outstanding gates are in [the run ledger](experiments.md).
