# F1 recurrent warp and training repeatability

## Why the same seed produced different checkpoints

A bilinear warp gathers samples from the previous frame. During backward
propagation several output pixels can send gradient to the same source pixel.
The shipping large scalar scatter uses floating-point atomic additions.
Scheduling changes their order, and f32 addition rounds at each step.

From the saved step 5,035, both minibatch losses and both forward recurrent
states repeated exactly in all 16 replays of each backend. On the last cursor,
the first backward tap, `f3.warp3`, returned different sums for identical
indices and source gradients. Across all 12 taps and 16 replays, 485 of
13,369,344 destination reads differed from ascending-order f32 on the shipping
kernel. The largest absolute difference from the f64 sum was 4.34×10⁻¹³.
Every destination on both implementations was within its f32 error bound.
The evidence identifies a reduction-order sensitivity. These measurements
support the numerical correctness of the original scatter on this batch.

| Replayed quantities | Shipping atomics | Fixed source order |
|---|---:|---:|
| Largest parameter-gradient change, 16 frozen replays | 5.82×10⁻¹⁰ | 0 |
| Model or Adam tensors differing after one normal update | 64 / 120 | 0 / 120 |
| Model or Adam tensors differing after two same-seed 12k fits | 120 / 120 | 0 / 120 |
| Logged losses differing across those 12k fits | 11,936 / 12,000 | 0 / 12,000 |

The ordered seed-1 repeats also match on every recurrent-state byte and
every row of the trainer's final evaluation. Their checkpoint SHA-256 is
`bf34917069f70f8ae34ccfc92a42d02fafdf60c946e58ca3e9d0415c5565f61f`.
This establishes repeatability for these two fits on this adapter and
software stack. Seeds 1 and 2 still sample different training trajectories.
The synthetic scalar scatter matched the ordered f32 reference, including
the 9- and 33-contributor fallback, on both RADV and LavaPipe.

## Experiment protocol

Replay update 5,036 from a saved eight-scene F1 training run. Both cursors
advance through moving frames 4–7. Fix the weights, Adam state, recurrent
state, sample order and GPU. Compare the shipping Meganeura gradient for
bilinear history sampling with a training-only version that sums contributions
in ascending source-pixel order. Check individual warp sums against a CPU
reference and compare complete saved updates by tensor name.

Train the ordered implementation twice with seed 1 and once with seed 2 for
12,000 updates from scratch, using the original 32×32 crops and four-frame
unroll. Compare with the first same-seed replays from the previous experiment.
Use the second existing seed-1 replay to measure sensitivity to the chosen
control. Evaluate the final weights with the same pinned inference executable
on ten development sequences. Score continuous history and resets every 16
frames, the locked crops, and paired whole-scene intervals (1,000 draws,
seed 31). Keep the previous advancement criteria for both seeds. Causal FLIP
must decrease, the lower bound on warm PSNR improvement must exceed zero,
and cold PSNR may lose at most 0.1 dB under either reset protocol.

## Follow-up on training data

With the summation order fixed, test whether F1 transfers better when its
training data include the five development case types at the same resolution.
Recover the first four training scenes per case from the archived 200-scene
manifest. Verify each capture, provenance sidecar and catalog selection by
SHA-256 before training. Fit the same two seeds and 12k schedule from scratch
and compare against the ordered eight-scene controls. This comparison changes
scene distribution, resolution, sequence length and the resulting cursor
schedule together. Use the same advancement criteria, development frames,
reset protocols and fixed pictures. These are development experiments.

Show the seed-1 native-resolution model beside its ordered eight-scene
control and the reference on development frames 0:0 (static), 2:31 (camera),
6:63 (lighting) and 8:31 (catalog). Show complete frames with the common
display transform. Use the original four 128×128 frames to check transfer
back to the smaller procedural scenes.

## Replaying a checkpoint

The GPU probe restores weights, optimizer moments, both recurrent states and
the saved sampler before every update. It reads the 12 backward warp outputs
for the last cursor, preserves the optimizer graph's normal fusion and keeps
intermediate buffers alive for readback. For each destination it sums the same
source values in ascending pixel order on the CPU, in both f32 and f64. The
roundoff check uses twice the contributor count times f32 epsilon times the
sum of absolute values, plus a tiny absolute floor. The synthetic case covers
0, 1, 2, 7, 8, 9 and 33 contributors, cancellation, zeros and invalid indices.

On the RX 7900 XT, use the checkpoint bundle's original captures in their
original order, including repeated paths. The synthetic-only invocation also
works on LavaPipe.

```sh
MEGANEURA_DEVICE_ID=0x744c cargo +1.92.0 run --release --locked \
  -p ommatidia-train --example warp-gradient -- \
  path/to/model.safetensors warp.json static.omd moving.omd static.omd moving.omd
VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.json \
  cargo +1.92.0 run --release --locked -p ommatidia-train --example warp-gradient -- \
  --synthetic-only synthetic.json
```

The experimental [Meganeura patch](../patches/meganeura-ordered-scalar-scatter.patch)
applies to revision `25ed41cc8426e98849ba6366d7da6634b0e2b57d`. It indexes
nonzero scalar contributions by destination, then sums each destination in
source order. For more than eight contributors it scans all sources. This
fallback prioritizes reproducibility. Its cost on other training workloads
has not been measured. The pinned inference executable used for scoring is
identical for every checkpoint.

To reproduce the ordered build, clone Meganeura at that revision beside a
fresh Ommatidia worktree and apply the patch inside the Meganeura checkout.
In the worktree's workspace dependencies point `meganeura` at
`{ path = "../meganeura" }`, regenerate the lockfile, and run the same
training or probe command there. Standard Ommatidia builds use the pinned
upstream dependency.
