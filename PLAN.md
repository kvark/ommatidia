# Plan: one end-to-end reconstruction model (config v4)

Status: adopted, 2026-09-27; Phases 0–1 complete, Phase 2 next. Owner: kvark.
Read all of §0–§3 before starting. Open decisions for the owner are in §12.

Implementation started from `origin/main` at `232a278`, after the owner merged
PRs #21 (this plan) and #22 (the rebaseline). No duplicate merge is needed.
`TASK.md` was deliberately removed; its closure and original gate are preserved
in [the archived sprint](docs/archive/quality-week.md). Run summaries go in
[docs/experiments.md](docs/experiments.md). Defaults D1–D6 apply.

## 0. Mission

Replace today's hand-tuned guide plus residual U-Net (config v3) with **one recurrent
network (config v4) that does denoising, 2× upscaling, anti-aliasing and temporal
accumulation itself**, then train it with a training loop that can actually saturate it.
Every tuning constant in today's reconstruction path disappears or moves into the
network: the à-trous pyramid and its fixed blend weights, history-length caps,
reactive/rejection thresholds, variance and moment features.

Why:

- v3's fixed guide makes the decisions that fail. Lighting lag comes from a fixed
  history weight: to reject stale history at h = 15/16 the network must emit residuals
  of order −150·(H−S)/(S+1). Blotches and cold-start noise come from a fixed four-level
  à-trous filter with 50% weight on its ~15-pixel level. The sprint diagnosed both
  (`docs/archive/quality-week.md`).
- The spatial path is the weak link. OIDN, single frame and no history, reached a
  smooth-crop ratio of 0.523 against v3's 0.552 on the first audit.
- Training is starved. Batch 1, about three visits per training frame per run, and
  ~300 ms per update against 1.2 ms of GPU inference per frame. One scene with 2,000
  cold updates went 28.1 → 32.6 dB; 40 scenes with 4,000 cold updates gained 0.13 dB.
  No capacity or architecture conclusion drawn at this budget is reliable.

**Definition of done**

1. v4 is the only model in the tree; v3 is archived (git tag plus a built executable).
2. v4 was trained from scratch with the new loop; `docs/experiments.md` lists every run.
3. v4 was evaluated once on the untouched confirmation set against the v3 control, and
   gates G1–G7 (§10) are reported as pass or fail, misses included.
4. README, `docs/design.md` and `docs/evaluation.md` describe v4, with images, videos,
   GPU time, memory and FLOPs per pixel, all reproducible from `main`.

Time box: about 12 working days. If a phase overruns its box by 50%, stop and report.

## 1. Design principles

**Learned by the one network**

- Spatial denoising and upsampling from low-resolution (LR) samples to the
  high-resolution (HR) output: support size, edge awareness, outlier handling.
- How much history to keep, per pixel and per lobe: accumulation length, disocclusion
  and lighting-change rejection.
- A latent recurrent state, replacing hand-made ages and moments.
- Anti-aliasing: references are anti-aliased; inputs are pixel-centre with projection jitter.

**Fixed structure** (geometry and physics, not tuning knobs)

- Bilinear warp of the previous state by renderer motion vectors. Taps outside the
  frame are dropped and the rest renormalized; no taps left means history is invalid.
  Hard reset on camera cuts.
- Re-modulation: `rgb = albedo·diffuse + specular + emission` (and specular albedo once
  it is captured, Phase 7).
- Output activations: `exp` for radiance, `sigmoid` for the history blend, `tanh` for
  the latent state.
- Per-frame exposure supplied by the host, applied as a pre-scale and removed after
  decoding.

**Forbidden in the reconstruction path:** new hand-tuned constants, such as filter
weights, history caps, thresholds, clamps or edge-stopping functions. If the model
seems to need a signal, add it as an input feature and let the network learn to use it.

## 2. Planning baseline (verified before adoption, 2026-09-27)

This section records the plan's starting assumptions. The adoption note above
supersedes its branch/merge status; implementation checks supersede test counts.

**Code**

- `main` (`e45ed32`) is stale. Its decoder applies the learned correction after history
  blending (`ommatidia/src/transport/graph.rs:177`), so the correction compounds through
  history. The steady-state gain is ≈ (1+0.1r)/(1−0.1·r·N), which diverges for
  r ≳ 0.3 at N = 32.
- `agent/rebaseline-2026-09-25` (`64102bc`, 29 commits ahead) holds the current work.
  [kvark/ommatidia#20](https://github.com/kvark/ommatidia/pull/20) was closed unmerged.
  The branch has:
  - the fixed decoder;
  - the Naga workgroup-layout patch (`patches/`, `scripts/prepare-naga.py`; upstream
    [gfx-rs/wgpu#9295](https://github.com/gfx-rs/wgpu/pull/9295) is still open);
  - evaluation and measurement tools: the OIDN reference (`oidn-reference`), GPU pass
    timing (`benchmark`), `score-sequences`, `export-references`, and crop scoring
    (`scripts/score-regions.py`);
  - fixes: the compact f16 loader, explicit device selection in GPU tests, and the
    f32/unfused inference policy;
  - the sprint documents `TASK.md` and `docs/quality-week.md`.
- The branch builds with a sibling `../blade` at `fbb4f28` after `python3
  scripts/prepare-naga.py`. 83 unit tests pass; 7 GPU tests are `#[ignore]`d.
- Meganeura (`ee3aea4`) already provides what this plan needs:
  - `set_grad_accumulate`/`zero_grad`: the mean gradient over K micro-batches;
  - `set_grad_clip_norm`;
  - `share_parameter_from`: sessions sharing weights;
  - `input_buffer`/`output_buffer`: direct GPU access;
  - `sigmoid`, `exp`, `clamp`, `softmax` (rows of a 2-D tensor), and `embedding` (a gather,
    used by today's differentiable warp);
  - a `batch` argument on `conv2d`.

**Data** (on the owner's workstation under `runs/quality-week-2026-09-26/`, not in git)

| Set | Content | Role |
|---|---|---|
| Training | 40 scenes × 64 frames, 1-spp 128×128 → 256×256, 1,024-spp targets, five cases (static, camera, objects, lights, catalog) | Training. The ordered capture list is in the recorded command of `cold-coverage-training/` |
| Development | 10 scenes × 64 frames, 4,096-spp | The only selection set |
| First audit | `docs/quality-benchmark.json` | Regression and diagnostics only; never tune on it |
| Confirmation | `docs/quality-confirmation.json`, seeds 810001–850001 | Locked, never evaluated. Phase 6 only |
| Diagnostics | `data/fit-train-long.omd` and `data/fit-long.omd` (scene 310001, independent input noise); `data/fit-dev-long.omd` (scene 310101) | Sanity fits |

**v3 reference numbers** (development set, 640 frames, unless noted)

| Model | PSNR | Cold PSNR | Notes |
|---|---:|---:|---|
| Original published runtime | 28.27 | — | The sprint's starting point |
| Frozen candidate `0d75f367…` | 30.17 | 26.10 | First audit 31.37 dB. Smooth-crop ratio 0.552 missed the ≤ 0.50 gate; per-frame ratios at frames 0/31/63 are 0.879/0.332/0.315 |
| Cold-coverage `89e81df0…` | 30.27 | 26.27 | Best v3 on development. **This is the v3 control** |
| OIDN (first audit) | 27.39 | — | Smooth-crop ratio 0.523; texture gradient error 3.64× |

**Performance** (v3 on the RX 7900 XT): 1.18 ms of GPU time per 128→256 frame
(automatic precision, fused) and 2.09 GFLOP per frame, 52% of it in the 232-channel
stem. Linear extrapolation to 540p→1080p is about 37 ms. Training takes ~300 ms per
update.

**Environment**

- GPU work (capture, training, evaluation, timing) needs the owner's workstation:
  RX 7900 XT, `--device-id 0x744c` or `MEGANEURA_DEVICE_ID=0x744c`.
- It also needs the isolated Mesa 26.2.3 driver; older Mesa silently drops catalog
  meshes (see `docs/catalog.md`).
- CPU-only machines can edit code, run unit tests, and run slow GPU tests on LavaPipe
  (`VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.json`).

## 3. Rules of engagement

- **One model.** No parallel architectures and no compatibility switches. The
  pre-registered fallbacks in Phase 5 replace parts of v4; they are never added as options.
- **Train from scratch.** No chains of warm starts. `--checkpoint` is only for resuming
  an interrupted run, with its optimizer state.
- **Splits.**
  - Select on development only.
  - Report the first audit, but never tune on it.
  - Evaluate the confirmation set exactly once, in Phase 6, after the freeze.
- **Evidence, lightweight.**
  - Wrap every run with `scripts/record-run.py`.
  - Add one row per run to `docs/experiments.md`, with at most two lines of notes.
  - No long prose logs.
  - The full freeze-and-hash protocol (`selection.json`, chronology checks) applies only
    to the final candidate.
- **Correctness before training.** That means unit tests, CPU/WGSL parity, the f64
  gradient check, the reload check, and zero Vulkan validation errors in debug on both
  RADV and LavaPipe.
- **Error bars.**
  - Every decision uses a bootstrap 95% confidence interval over sequences.
  - Add a second seed when the interval straddles a threshold.
  - Spend no more than half a day on a question worth less than 0.1 dB.
- **Git.** Commit and push only to the working feature branch. The owner creates
  PRs and performs merges. Never push directly to `main`, create PRs, or merge them.
- **Stop and report to the owner** on any of these:
  - the Phase 1 profiling finding in R1;
  - decision point A;
  - any final gate miss;
  - a time-box overrun;
  - anything that touches data licensing (HSSD, DTC).

## 4. Phase 0: consolidate (½ day)

Goal: a correct `main` to build on.

- [x] Close the quality sprint using the removed `TASK.md`'s "Miss" rule:
  - Record the outcome of the width-32 run (`wide-cold-training/`) if it finished;
    otherwise stop it and record that.
  - Record at most 15 closure lines in the archived sprint: gates missed, the
    strongest evidence, and "next hypothesis: `PLAN.md`". Do not restore `TASK.md`.
  - Move `docs/quality-week.md` to `docs/archive/`.
  - Run no further sprint experiments.
- [x] Consolidate the rebaseline and plan into `main`: the owner merged PRs #21
  and #22 before implementation began (rebased, not duplicated with a merge commit).
- [x] README consistency (decision D1, default shown):
  - Keep the published checkpoint.
  - Note that its numbers came from the runtime before the decoder fix.
  - Archive that runtime so the numbers stay reproducible, as the sprint's
    `initial-linear-build/` did from `049a7bd`.
- [x] Push a tag `archive/experiments-2026-09-25` → `e0922c6`. At planning time that commit was
  reachable only through PR #20's ref. Update the pointers in README and `docs/design.md`.
- [x] Naga: prepare the in-tree patch's handling of SPIR-V below 1.4 for
  gfx-rs/wgpu#9295. Keep the tested `patches/` and Cargo pins until a pinned
  upstream revision contains the fix.
  - Local port at `target/naga-pr9295` (base `731fd872`): targeted SPIR-V
    validation, all-feature Clippy and the Naga suite pass. See the
    [review handoff](docs/archive/naga-upstream.md); no upstream branch has changed.
  - **Owner decision, 2026-09-27:** upstream publication is deferred as a
    non-blocking follow-up, not a Phase 0 completion gate. Human review is still
    required before any later publication. The local fix and correctness gates
    remain in force; publication is not a prerequisite for training.

Done when: `main` builds with the documented steps; fmt, clippy, unit tests and CI
(including the LavaPipe job) are green; and every README number is reproducible from
archived code.

Completed 2026-09-27: the owner merged [PR #23](https://github.com/kvark/ommatidia/pull/23)
into `main` at `a153c1e`. All five [post-merge CI jobs](https://github.com/kvark/ommatidia/actions/runs/36298665573)
pass, including LavaPipe. The merged tree matches the tested PR; the recorded
release build and exact archived-result reproductions remain valid.

## 5. Phase 1: measure, and fix the evaluation contract (1 day)

Goal: know where training time goes, and have the metrics that will decide v4.

- [x] **Profile the v3 training loop.** Time each part:
  - frame decode;
  - the warm-up `advance` loop;
  - per-slot `advance`;
  - `read_prepared` (readback and `history_maps`);
  - `feed` uploads;
  - `session.step()` plus `wait`;
  - loss readback.

  Run 200 updates on the 40-scene corpus on the RX 7900 XT and put the table in the PR.
  **If `step()` plus `wait` alone exceeds 50% of the time, report to the owner before
  Phase 3**: the bottleneck is then inside Meganeura, not the loop.
  Measured: 438.30 ms/update; readback/history maps 63.00%, step/wait 3.69%.
  R1 not triggered. [Table for the owner's PR](docs/training-profile.md).
- [x] **FLOPs.** Add `Network::macs()` in `ommatidia/src/neural.rs` (sum the conv
  multiply-accumulates from shapes). Print it at startup and store it in `training.json`.
  Check that v3 at width 16 on a 128×128 LR grid gives 1.05 G multiply-accumulates
  (2.09 GFLOP) per frame.
- [x] **Evaluator** (`ommatidia-train/src/bin/transport.rs`):
  - `--reset-every N`: simulated cuts during evaluation; off by default.
  - Standard development protocol = two runs: causal, and `--reset-every 16`.
  - Add a `frames_since_reset` column to `frames.csv`. Aggregate cold (0 frames since
    reset), early (1–7) and warm (≥ 16) separately.
  - `--control-run DIR`: read a previous run's `--save-linear` outputs (same dataset
    order; references must match byte for byte) and report per-frame deltas. This
    replaces the zero-head guide baseline once v3 is deleted.
  - During training, evaluate causal metrics only, without PNGs, every 10,000 steps.
    Candidates get the full protocol.
- [x] **ꟻLIP.** Report the mean LDR ꟻLIP per frame on the fixed display transform
  (`x/(1+x)`, then sRGB). Use the official implementation in a scoring script
  (`pip install flip-evaluator`) or port it to Rust. Validate against the reference
  implementation's test images to within 1e-4.
- [x] **Confidence intervals.** Write a script that compares two runs' `frames.csv`: mean
  difference and 95% bootstrap interval (1,000 resamples of sequences). Cover PSNR,
  ꟻLIP, temporal MSE, energy, and the cold/early/warm splits.
- [x] **Development crops.** Commit the development rectangles the sprint used (recorded
  in `dev-*-{middle,cold}.json` under `runs/`) as `docs/dev-crops.json`, using the schema
  of `quality-benchmark.json`. If they cannot be recovered, define new ones from
  references only (`export-references`).
  Recovered all 24 region/frame selections exactly; source hashes and the historical
  coverage limits (frames 0/31, five of ten sequences) are in `docs/dev-crops.json`.
  Owner approved an early extension: reference-only inspection adds frame 3 to
  the sixteen unchanged rectangles, before the complete protocol evaluations.
- [x] **v3 control.**
  - Build and archive the v3 release runtime (`runs/archive/v3-runtime/`, with a recorded
    build manifest).
  - Evaluate `89e81df0…` on development, causal and `--reset-every 16`, with `--save-linear`.
  - Start `docs/experiments.md` with this row.
  - **Run nothing on the confirmation set.**

Done when: the profile table exists, and the development protocol produces cold, early
and warm PSNR, ꟻLIP, temporal MSE, energy, crop ratios and confidence intervals for the
v3 control.

Completed 2026-09-27: [measurements and coverage](docs/phase1-results.md),
[full-precision results and CIs](docs/phase1-results.json). The 200-update profile
does not trigger R1. Both development protocols cover all 640 frames; 40 crop/frame
selections include the owner-approved reference-only early extension. The control
runtime, source and unchanged `89e81df0…` weights are archived. Exact reproduction,
reference identity, reset strata, intervals and crop coverage passed the recorded
artifact audit. Confirmation was not evaluated. No v4 training or promotion yet.

## 6. Phase 2: the v4 model (3 days)

Goal: the single model, numerically verified, replacing v3.

Frame step. Per HR pixel; the two lobes are diffuse illumination and specular radiance.

```text
e       = frame.exposure                         # host input, per frame
valid   = !reset && any in-frame warp tap        # geometry, not a heuristic
H, S, P = warp(prev.{lobes, latent, normal_depth}, motion)   # zeros where !valid
x = concat(
      c(e·samples), lr_normal_depth, jitter, subpixel_offsets,      # LR grid
      packed(hr_normal, hr_depth, albedo, F0, roughness, motion),  # HR, 4 slots
      packed(c(e·H), S, P, valid))                                 # c(v) = v/(1+v)
f       = unet(x)                  # neural.rs encode(); add biases
z, a, s = heads(f)                 # 1×1 convs: 6, 2 and C_s channels per HR pixel
spatial = exp(clamp(z, -16, 11)) / e       # direct estimate per lobe and channel
alpha   = sigmoid(a) · valid                # per lobe
lobes   = alpha·H + (1 − alpha)·spatial
latent  = tanh(s)
rgb     = albedo·lobes.diffuse + lobes.specular + emission
next    = {lobes, latent, normal_depth, albedo}
```

Defaults: C_s = 4, width 16, three levels, unroll 4. Initialize the `z` bias to the log of
the corpus-mean exposure-normalized lobe radiance, and the `a` bias to logit(0.8). Other
head weights start at zero. The stem shrinks from 232 to about 134 input channels, so
expect about 1.6 GFLOP per frame against v3's 2.09.

Direct prediction is the default because it is the simplest form of "one model", and
OIDN shows direct U-Nets denoise well given enough capacity and data. Kernel prediction
is fallback F1 in Phase 5, used only if cold frames miss decision point A.

- [ ] **Config v4** (`ommatidia/src/transport/mod.rs`):
  - Drop `diffuse_frames` and `specular_frames`; add `latent_channels` and `levels`.
  - `Frame` gains `exposure: f32`. `Surface` gains F0; `SpecularF0` is already captured
    at LR and HR.
  - `State` becomes lobes + latent + normal/depth + albedo.
  - Reject v3 configs with a clear error; no compatibility mode.
- [ ] **Graph** (`ommatidia/src/transport/graph.rs`):
  - A new `build()` following the frame step.
  - Keep the warp inside the graph for training and inference, as a gather via
    `embedding` the way `warp()` does today.
  - Keep today's loss terms (`LossWeights`); the temporal loss spans the unroll through
    the differentiable warp of lobes and latent.
  - Add graph outputs for the final unrolled frame's lobes and latent, which Phase 3
    needs to carry state; confirm Meganeura allows loss plus extra outputs in training
    mode.
  - Add α as an optional debug output.
- [ ] **Runtime.**
  - `prepare.wgsl` shrinks to `pack` (features plus warp indices and coefficients,
    computed on the GPU) and `resolve`.
  - Delete `seed`, `atrous`, moments, ages, prior and candidates.
  - `native.rs` keeps its host API shape (`record_prepare`, the session step,
    `record_resolve`, `process`), plus exposure.
- [ ] **CPU reference** (`ommatidia/src/transport/cpu.rs`): scalar pack, warp and resolve
  that match the WGSL.
- [ ] **Tests.** Adapt `ommatidia/tests/transport.rs`:
  - CPU/WGSL parity over features, warp, a recurrence of 24 or more frames, reset and HDR;
  - the f64 check of the loss and every parameter's gradient, fused and unfused;
  - bit-exact reload.

  Add unit tests:
  - α = 0 gives exactly the spatial estimate;
  - `valid = 0` ignores history;
  - radiance ×k with exposure ×1/k gives output ×k within 1e-5.
- [ ] Tag the last v3 commit `archive/v3-guide-residual`, then delete the v3 guide,
  decoder and zero-head baseline. Update the LavaPipe CI smoke (capture, train, reload)
  for v4.
- [ ] **Smoke test.** Run 2,000 updates of the existing trainer (full frame, batch 1) on
  the single-scene diagnostic. Loss must fall, outputs stay finite, and a reload
  reproduces results exactly.
- [ ] Rewrite `docs/design.md` for v4, at most one page.

Done when: all tests pass (regular, plus GPU tests on RADV and LavaPipe with zero
validation errors), no v3 code remains, and FLOPs per pixel are reported.

## 7. Phase 3: training loop v2 (2 days)

Goal: at least 10×, ideally 20×, today's pixel-gradients per second, with mini-batches
that mix cold and warm windows and no CPU round trips.

Design:

- **Cursors.** B = 8 persistent cursors, each a (sequence, crop, frame) tuple.
  - Crops are 64×64 LR (128×128 HR) and fixed for a cursor's lifetime.
  - Each cursor keeps its state (lobes, latent, normal/depth, albedo) on the GPU.
  - Warp taps that fall outside the crop are invalid, as at the frame border.
  - Exclude a 4-pixel HR margin from the losses.
- **Step.**
  1. For each cursor, unroll k = 4 frames with gradients. The first frame's history is
     the cursor's state, detached.
  2. Accumulate gradients over the B cursors (`set_grad_accumulate(B)`, with
     `zero_grad()` before each step), then take one Adam step.
  3. Copy the final frame's lobes and latent into the cursor's state on the GPU (Blade
     transfer pass; `output_buffer`/`input_buffer`).
  4. If utilization is low because of dispatch overhead, move to a true batch dimension
     (`conv2d` already takes `batch`).
- **Resets.**
  - A cursor respawns cold (new sequence, crop and start frame) at the end of its
    sequence or after a random lifetime.
  - Each window additionally resets with probability 0.1.
  - Aim for 10–20% cold loss windows and log the actual fraction.
- **Augmentation.** Scale radiance by 2^U(−2, 2) per cursor, on inputs and targets alike
  (exposure stays 1), constant over the cursor's life. No geometric flips.
- **I/O.**
  - Records are fixed-size: memory-map the `.omd` files and read only the crop.
  - Prefetch the next step's crops on a worker thread.
  - Measure I/O; if the corpus exceeds RAM and reads stall, pre-cut it into tiles.
- **Weights.** The evaluation session shares parameters with the training session
  (`share_parameter_from`), so there are no per-update copies.
- **Optimizer.**
  - Adam with a 500-step linear warm-up, then cosine decay to 10%.
  - Clip the gradient norm at 1.0.
  - Checkpoint every 5,000 steps, including Adam state, for resume.

Tests:

- [ ] Accumulating B identical micro-batches equals one micro-batch's gradient;
  accumulating B different ones equals the mean of their separate gradients.
- [ ] State carry: running T frames as T/k carried steps, with no gradient, matches one
  long causal inference within 1e-5.
- [ ] Taps outside the crop are invalid, and the loss margin is applied.
- [ ] Report steps per second and pixel-gradients per second against the Phase 1 baseline.

Done when: the tests pass; throughput is at least 10×; a 5,000-step run on the 40-scene
corpus finishes with a development evaluation; and the CI smoke and reload still pass.

## 8. Phase 4: more training scenes (in parallel with Phase 2; about a day of GPU time)

Goal: enough scenes that a 60,000-step run does not memorize.

- [ ] Capture 160 more training scenes (32 per case) × 64 frames using the exact recorded
  capture commands of the 40-scene corpus, changing only seeds. For example, use base
  seeds 1,010,001–1,050,001 in steps of 10,000.
- [ ] Before capturing, check seed disjointness against every capture: training,
  development, both audits, diagnostics and published ancestry, as the sprint did.
- [ ] Catalog scenes use the existing 22-family training pool; verify visibility over
  each full trajectory.
- [ ] Expect about 10,000 frames, roughly 40 GB of f16 at 256² HR. This needs the Phase 3
  loader.
- [ ] Leave the development set unchanged.

Done when: there are 200 training scenes, with membership, hash and visibility checks
recorded.

## 9. Phase 5: training ladder and decision point A (3 days)

Every rung reports, in one `docs/experiments.md` row per run:

- development metrics under both protocols;
- confidence intervals against the v3 control;
- α maps and histograms for three fixed development frames;
- FLOPs per pixel.

1. **Sanity.** Train from scratch on a single scene (`fit-train-long.omd`) for at most
   10,000 steps. On the independent-noise stream (`fit-long.omd`) it must come within
   1 dB of v3's single-scene results: 32.6 dB resetting every frame, 33.6 dB causal. If
   it does not, debug the model or the loop, not the data.
2. **Budget curve.** 40 scenes, width 16, seed 1, 60,000 steps.
   - If development metrics still improve at the end, the model is budget- or
     data-bound: use 200 scenes from rung 3 on.
   - If development metrics flatten while training loss keeps falling, it is overfitting:
     add scenes.
3. **Capacity.** On 200 scenes, try {width 16, 32} × {3, 4 levels}, one seed each at the
   rung-2 budget. Give the best two a second seed. Prefer the cheaper configuration
   unless the larger one wins the cold smooth-crop ratio by at least 10% with an
   interval excluding zero.
4. **Decision point A.** Best v4 against the v3 control on development, each with its
   own history. All of these must hold:
   - cold smooth-crop MSE ratio ≤ 0.80;
   - warm PSNR within 0.1 dB of v3 or better (interval lower bound ≥ −0.1 dB), and warm
     smooth-crop ratio ≤ 1.0;
   - temporal MSE ratio ≤ 1.0, and the lighting sequences' final frames no worse than v3;
   - energy ratio within [0.98, 1.02], and ꟻLIP ≤ v3.

   **Pass** → Phase 6. **Fail** → one pre-registered fallback at the same budget, as a
   replacement:
   - **F1**, if cold or spatial quality fails: add a kernel-prediction path (softmax
     weights over a 5×5 LR neighborhood of exposure-normalized samples), mixed with the
     direct estimate by a learned per-pixel weight.
   - **F2**, if lag or flicker fails: per-channel α plus a signed history-correction term.
   - **F3**, if energy fails: raise the linear and low-frequency loss weights, and
     compute the loss in log space.

   Still failing after one fallback → stop and report the evidence to the owner.

## 10. Phase 6: freeze, final evaluation and publication (1–2 days)

- [ ] Freeze the selected checkpoint with the existing selection protocol (the
  `docs/results/selection.json` schema: weights, config, executable, source,
  timestamps) **before** any confirmation run.
- [ ] Evaluate v4, the v3 control (archived runtime) and OIDN (`oidn-reference`) on the
  confirmation set (its first use), the first audit and development.
  - Run both protocols with `--save-linear`.
  - Score crops with `scripts/score-regions.py --selection` and sequences with
    `score-sequences`.
  - Add ꟻLIP and confidence intervals.
- [ ] **Gates** (v4 against the v3 control, on confirmation):
  - **G1:** cold smooth-crop ratio ≤ 0.80.
  - **G2:** warm smooth-crop ratio ≤ 1.0, and edge and texture gradient ratios ≤ 1.05.
  - **G3:** temporal MSE ratio ≤ 1.0, with no sequence above 1.10.
  - **G4:** PSNR non-inferior (interval lower bound ≥ −0.1 dB) and ꟻLIP ≤ v3.
  - **G5:** energy within ±2%.
  - **G6:** correctness: all tests pass, with zero validation errors.
  - **G7:** performance is reported, not gated. Report:
    - GPU milliseconds per frame under both the f32-unfused and the automatic f16 policy;
    - memory;
    - FLOPs per pixel;
    - the 1080p extrapolation.

    More than 2× v3's FLOPs per pixel needs owner approval.
  - Also report the archived sprint's original ≤ 0.50 smooth-crop gate against
    the original published checkpoint for continuity.
- [ ] Measure performance with `benchmark` on an idle GPU: two warm-up and three measured
  sequences.
- [ ] Publish:
  - README images: frame 31 of sequences 0, 2 and 8. Videos: sequences 0, 2, 4, 6 and 8,
    rendered with `scripts/render-comparisons.py`.
  - Update `docs/results/`, `docs/design.md` and `docs/evaluation.md`.
  - State every gate outcome, including misses.

## 11. Phase 7: inputs v2 (after Phase 6, separate PR)

Each item changes the same model's inputs. Train each from scratch and compare on
development.

- **Specular albedo.** Add a `SpecularAlbedo` plane: pre-integrated specular directional
  albedo from F0, roughness and N·V, using the standard analytic fit. It goes in the
  G-buffer capture (`ommatidia-data/src/gbuffer.{rs,wgsl}`). Demodulate and re-modulate
  the specular lobe with it.
- **Expected previous-frame depth** (`Surface.motion.z`), computed from the capture's
  camera and object transforms. Today it is never populated.
- **Reflection motion** (`Surface.specular_motion`, also never populated) or specular
  hit distance. This needs a change to Blade's path tracer (decision D5). Then warp a
  second specular history, and turn the blend into a softmax over {spatial,
  surface-warped, reflection-warped}.
- **Recapture** development and confirmation with the same seeds. Every previously locked
  plane must match byte for byte, and only the new planes may be added; record that
  verification.

## 12. Open decisions for the owner

Defaults apply unless the owner changes them. Gate thresholds are frozen once Phase 5
starts.

| # | Decision | Default |
|---|---|---|
| D1 | README before the pivot | Keep the published checkpoint; note the pre-fix runtime; archive it |
| D2 | HSSD interiors in training (the checkpoint would inherit CC BY-NC) | No |
| D3 | Input noise: independent 1-spp paths or ReSTIR (Blade can capture both) | Keep 1-spp paths; revisit after Phase 6 |
| D4 | Training framework | Stay on Meganeura; revisit if R1 triggers |
| D5 | Blade changes for reflection motion or hit distance | After Phase 6 |
| D6 | Thresholds in Phases 5 and 6 | As written |

## 13. Risks

- **R1.** Meganeura's training step itself may dominate time, so loop changes cannot
  reach the throughput goal. Phase 1 detects it; report kernel-level hotspots to the
  owner.
- **R2.** v4 trained from scratch starts behind v3, because it has no guide prior. The
  ladder and fallback F1 cover this; do not conclude anything below the rung-2 budget.
- **R3.** A learned α may lag or flicker. Guard with the lighting cases, the temporal
  loss, and α maps and histograms at every evaluation.
- **R4.** Energy may drift with direct prediction. Watch the energy ratio, keep the
  linear loss term and initialize the output bias; F3 is the fallback.
- **R5.** Crop borders may leave artifacts. Apply the loss margin; development
  evaluation is always full-frame.
- **R6.** Data volume exceeds RAM. Use memory-mapped crop reads, or pre-cut tiles.

## 14. Out of scope

DLSS-parity claims; 1080p integration; attention or transformer backbones; frame
generation; new asset sources; any model other than v4.

## Appendix: commands

```sh
python3 scripts/prepare-naga.py          # patched Naga into target/patched-naga
cargo +1.92.0 fmt --all -- --check
cargo +1.92.0 clippy --workspace --all-targets --locked -- -D warnings
cargo +1.92.0 test --workspace --locked
python3 scripts/test-score-regions.py && python3 scripts/verify-results.py

# GPU tests in debug with validation, on the RX 7900 XT or LavaPipe
MEGANEURA_DEVICE_ID=0x744c python3 scripts/record-run.py runs/<name> -- \
  cargo +1.92.0 test -p ommatidia --test transport --locked -- \
  --ignored --nocapture --test-threads=1

# Evaluation with full-precision outputs
python3 scripts/record-run.py runs/<name> -- target/release/transport --eval-only \
  --checkpoint <ckpt> --eval-data <dev...> --out runs/<name>/eval --save-linear \
  --device-id 0x744c

# Crop scoring against a control run
python3 scripts/score-regions.py --benchmark docs/quality-confirmation.json \
  --before-run <control> --after-run <candidate> \
  --selection docs/results/selection.json --out <report.json>
```
