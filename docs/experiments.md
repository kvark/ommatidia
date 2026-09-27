# v4 run ledger

Plan: [PLAN.md](../PLAN.md). Paths are relative to the repository. Each new run
uses `scripts/record-run.py`; its manifest, log and inputs remain under `runs/`.
Only development selects models. Confirmation is reserved for Phase 6.

| Phase | Run | Budget / purpose | Outcome |
|---|---|---|---|
| 1, control | `v4-phase1/v3-control-causal-complete` | Frozen `89e81df0…`, 640 development frames; 40 predeclared crop selections | 30.270709 dB; 10 resets, 630 temporal pairs. All PNGs/float outputs reproduce the first run exactly. |
| 1, control | `v4-phase1/v3-control-reset16-complete` | Same control/data; reset every 16 frames | 29.607780 dB; 40 resets, 600 temporal pairs. All PNGs/float outputs reproduce the first run exactly. |
| 2 | `v4-phase2/initial-check` | Workspace/all-target compile after v4 replacement | Pass; initial library-only compile also passed before recording. |
| 2 | `v4-phase2/initial-unit` | First v4 unit run | Two fixture failures: missing U32 feeds and Blade-assigned shader bindings. Fixed without changing the model. |
| 2 | `v4-phase2/unit` | Workspace unit suite | 91 pass; 132 stem channels, 174,576 parameters, 1.62949 GFLOP/frame. |
| 2 | `v4-phase2/radv-recurrence` | Independent CPU/f64 vs GPU, 24 recurrent frames | Pass including reset/HDR and packed warp maps; zero validation errors. |
| 2 | `v4-phase2/clippy` | Warnings denied | One iterator-style error; fixed. |
| 2 | `v4-phase2/clippy-fixed` | Workspace/all targets | Pass. |
| 2 | `v4-phase2/radv-gpu` | Seven debug GPU gates | Four inference gates pass; three training gates expose inference-only Clamp in pinned Meganeura. |
| 2 | `v4-phase2/lavapipe-gpu` | Same seven debug GPU gates | Same Clamp limitation; zero validation errors. |
| 2 | `v4-phase2/radv-gpu-differentiable-clamp` | ReLU clamp, gradient and reload gates | Five pass; f64 helper's error bounds overflow, and small production probes hit f32 loss resolution. No training run started. |
| 2 | `v4-phase2/lavapipe-gpu-differentiable-clamp` | Same checks on software Vulkan | Same two failures; learning/reload passes, zero validation errors. |
| 2 | `v4-phase2/radv-f64-finite-bounds` | Exact piecewise clamp; fixed finite f64 tolerances | Every parameter gradient and all carried outputs pass, fused/unfused; finite differences pass; zero validation errors. |
| 2 | `v4-phase2/radv-production-resolved-probes` | Production-size probes with loss-ULP-aware step sizes | All tensors pass two probes; convergence/agreement tolerances unchanged. |
| 2 | `v4-phase2/lavapipe-finite-bounds` | Full GPU suite with finite tolerances | Seven pass, including every parameter gradient and carried output; zero validation errors. |
| 2 | `v4-phase2/radv-final-gpu` | Full GPU suite, RX 7900 XT | Seven pass; zero validation errors. |
| 2 | `v4-phase2/build-debug` | Debug workspace | Pass. |
| 2 | `v4-phase2/build-release` | Release workspace | Pass. |
| 2 | `v4-phase2/verified-reload-build` | Debug workspace with final live/loaded parameter check | Pass. |
| 2 | `v4-phase2/verified-reload-release-build` | Release workspace with final live/loaded parameter check | Pass; executable used by the 2,000-update diagnostic. |
| 2 | `v4-phase2/final-unit` | Workspace unit suite including clamp derivative/extreme logits | 92 pass. |
| 2 | `v4-phase2/final-clippy` | Workspace/all targets, warnings denied | Pass. |
| 2 | `v4-phase2/capture-train` | Fresh two-frame 16→32 debug LavaPipe capture, scene 7 | Pass; F0 and split radiance present, zero validation errors. |
| 2 | `v4-phase2/capture-dev` | Independent capture, scene 700 | Pass; zero validation errors. |
| 2 | `v4-phase2/lavapipe-evaluation-smoke` | Fresh captures: train, periodic/final metrics, reload, saved control and cuts | Pass; exact reload, zero self-control deltas, mismatch rejection; zero validation errors. |
| 2 | `v4-phase2/python-checks` | Catalog/crop/bootstrap/render tests and gallery verifier | 21 pass; historical gallery, crop selection and confirmation hashes unchanged. |
| 2 | `v4-phase2/flip-reference-tests` | Official FLIP fixtures | Four pass; mean within the fixed 1e-4 tolerance. |
| 2 | `v4-phase2/flip-without-baseline` | v4 learned-only evaluation with saved self-control | Pass; learned/control FLIP identical, no baseline files required. |
| 2 | `v4-phase2/single-scene-2000` | From scratch, default width/latent/levels, four-frame BPTT, batch one | 2,000 updates, 421.10 s; loss 0.014448 → 0.000405, all parameters finite and bit-exact after reload. Different diagnostic scene: 22.338 dB; not promoted. |
| 2 | `v4-phase2/lavapipe-ci-train-reload` | Exact eight-update CI trainer settings on fresh captures | Pass; reloaded CSV byte-identical, zero validation errors. |
| 2 | `v4-phase2/final-format` | Rust formatting on committed implementation | Pass. |
| 2 | `v4-phase2/single-scene-reload` | New process, final diagnostic checkpoint, 64 different-scene frames | CSV and diagnostics byte-identical to final training evaluation; finite float outputs. |
| 2 | `v4-phase2/single-scene-independent-noise` | Same fitted scene, independent input-noise capture; 64 frames | 36.147 dB, cold 35.506; not evidence of generalization. Every PNG and float output retained. |
| 2 | `v4-phase2/verify-smoke` | Audit all updates, cold/warm loss windows, reload, outputs and protected hashes | Pass; first/last 100-update mean 0.005466 → 0.000416, 64 exact reload frames, 256 finite float images. Confirmation/gallery/crop hashes unchanged. |
| 0, historical | `quality-week-2026-09-26/wide-cold-training` | 4,000 updates, width-32 v3 | Complete, rejected: development 29.385 dB vs parent 30.271; cold smooth ratio 0.820 vs original. |
| 0, historical | `quality-week-2026-09-26/balanced-exposure-training` | 16,000 planned; 9,277 recorded | Interrupted, not resumed. Last complete eval at 8,000: 30.061 dB, cold/middle smooth ratios 0.941/0.468 vs original; not promoted. |
| 0 | `v4-phase0/fmt` | Rust 1.92 formatting | Pass. |
| 0 | `v4-phase0/clippy` | Workspace/all targets, warnings denied | Pass. |
| 0 | `v4-phase0/unit` | Workspace Rust tests | 83 pass; eight GPU checks run separately. |
| 0 | `v4-phase0/release-workspace-build` | Rust 1.92, locked release workspace | Pass; runtime/build/test sources match `origin/main` at `232a278`. |
| 0 | `v4-phase0/lavapipe-tests` | Eight debug GPU checks | Eight pass, zero validation errors. |
| 0 | `v4-phase0/radv-tests` | Eight debug GPU checks, RX 7900 XT | Eight pass, zero validation errors. |
| 0 | `v4-phase0/python-tests` | Catalog/crop/render tests and README verifier | 14 tests pass; published evidence agrees. |
| 0 | `v4-phase0/archive-published` | Preserve the pre-fix executable/build/weights | Ten files copied with checked hashes; originals retained. |
| 0 | `v4-phase0/published-reproduction` | Both published checkpoints, both fresh audit sets | 512 frame evaluations; metrics exact, 1,536 PNGs byte-identical. |
| 0 | `v4-phase0/published-regression-reproduction` | Both checkpoints, older published regression sets | 256 frame evaluations; metrics exact, 768 PNGs byte-identical. |
| 0 | `v4-phase0/lavapipe-smoke` | Capture, eight training updates, reload | Pass; seven PNG/score files byte-identical, zero validation errors. |
| 0 | `v4-phase0/cache-check` | Cold/warm texture-cache independence | Captures byte-identical, zero validation errors. |
| 0 | `v4-phase0/naga-port-test` | Upstream PR #9295 port, SPIR-V 1.3/1.4, three init modes | Two tests pass, including `spirv-val`; owner deferred upstream publication. |
| 0 | `v4-phase0/naga-port-clippy` | Naga all-target/all-feature lint | Pass, warnings denied. |
| 0 | `v4-phase0/naga-port-suite` | Full Naga tests | 139 unit + 232 integration pass; two snapshot tests fail because `spirv-cross` is absent. |
| 0 | `v4-phase0/naga-port-gpu-suite` | Full upstream `cargo xtask test` | Could not run GPU tests: `cargo-nextest` is absent. |
| 0 | `v4-phase0/naga-port-suite-with-cross` | Full Naga tests with isolated SPIRV-Cross | 139 unit, 234 integration and six doctests pass; validator test was run separately. |
| 0 | `v4-phase0/naga-snapshot-baseline` | Same snapshot generator on unmodified PR head `731fd872` | Three tests pass; isolates five SPIRV-Cross-version-only sidecar changes. |
| 0 | `v4-phase0/naga-snapshot-check` | Compare snapshots; assemble/validate changed SPIR-V | Only three shader snapshots and their sidecars change; all three modules pass `spirv-val`. |
| 0 | `v4-phase0/naga-port-gpu-with-nextest` | Full upstream GPU suite with isolated test tools | 1,915 pass, nine fail, 19 skipped: eight need DXC; one OpenGL expected failure unexpectedly passes. |
| 0 | `v4-phase0/naga-gpu-failures-baseline` | Rerun failing families on unmodified PR head `731fd872` | 27 pass and the same nine fail; no newly introduced failure in these families. Full suite is not green. |
| 1 | `v4-phase1/profile-unit` | Profile accounting and MAC tests | One new assertion failed: training MACs also include the low-frequency-loss convolution; corrected the assertion. |
| 1 | `v4-phase1/profile-unit-fixed` | Workspace tests | 86 pass; eight GPU checks run separately. |
| 1 | `v4-phase1/profile-build` | Release trainer | Pass; source snapshot omitted the then-untracked profiling module. Re-recorded below. |
| 1 | `v4-phase1/profile-build-complete-source` | Release trainer, full tracked source diff | Pass; complete profiling source retained. |
| 1 | `v4-phase1/profile-radv` | Eight debug GPU checks, RX 7900 XT | Eight pass, zero validation errors. |
| 1 | `v4-phase1/profile-lavapipe` | Eight debug GPU checks | Eight pass, zero validation errors. |
| 1 | `v4-phase1/profile-clippy` | Workspace/all targets, warnings denied | Pass. |
| 1 | `v4-phase1/crop-recovery-tests` | Crop schema and exact historical recovery | Ten pass; all 24 crop/frame selections match. |
| 1 | `v4-phase1/profile-200` | 200 updates, original 40 scenes, RX 7900 XT | 2.282 updates/s; readback/history maps 63.00%, step/wait 3.69%; R1 not triggered. [Table](training-profile.md). |
| 1 | `v4-phase1/eval-contract-unit` | New evaluator contract tests | Compile failed on an ambiguous closure index; added its `usize` type. |
| 1 | `v4-phase1/eval-contract-unit-fixed` | Evaluation/library unit tests | 15 pass, including ordered control identity, byte references and missing strata. |
| 1 | `v4-phase1/eval-contract-clippy` | Workspace/all targets | Pass, warnings denied. |
| 1 | `v4-phase1/flip-environment` | Isolated Python 3.12, official FLIP 1.7 / NumPy 2.2.6 | Installed pinned CPU scoring dependencies. |
| 1 | `v4-phase1/flip-fixtures` | Pinned official reference images, test source and license | Downloaded from NVlabs revision `b475eb4`; hashes retained in the reference test. |
| 1 | `v4-phase1/bootstrap-tests` | Whole-sequence bootstrap contract | Five tests pass, including correlated clusters, nulls, unequal temporal counts and mismatches. |
| 1 | `v4-phase1/flip-reference-tests` | Official mean plus extra bit-exact magma check | Mean passes 1e-4; extra visualization equality fails (0.76% of channels, max 3/255). |
| 1 | `v4-phase1/flip-mean-reference-tests` | Prescribed mean tolerance; magma differences diagnostic | Four tests pass; mean 0.159714609385 vs 0.159691 (error 2.36e-5). |
| 1 | `v4-phase1/eval-contract-build` | Debug evaluator | Pass. |
| 1 | `v4-phase1/eval-contract-smoke` | LavaPipe training/reload/cut/control integration | Pass; no training PNGs, zero self-control deltas, reset mismatch rejected; zero validation errors. |
| 1 | `v4-phase1/eval-contract-final-unit` | Workspace unit tests | 92 pass; eight GPU checks previously passed on both backends. |
| 1 | `v4-phase1/eval-contract-final-clippy` | Workspace/all targets | Pass, warnings denied. |
| 1 | `v4-phase1/eval-python-tests` | Bootstrap, crop, catalog/render tests and published evidence | 21 tests pass; published evidence unchanged. |
| 1 | `v4-phase1/flip-smoke` | Score raw GPU outputs including saved control | Pass; learned and self-control FLIP identical. |
| 1 | `v4-phase1/bootstrap-smoke` | Paired control/learned enriched CSVs | All defined differences and interval endpoints exactly zero; unsupported strata null. |
| 1 | `v4-phase1/eval-fmt` | Rust formatting | Pass. |
| 1 | `v4-phase1/v3-release-build` | Locked release control runtime, clean `cf37c0a` | Pass; build manifest retained with source/dependency revisions. |
| 1 | `v4-phase1/archive-v3-control` | Preserve corrected v3 source, executable, build evidence and weights | Eleven files hash-checked in `runs/archive/v3-runtime/`; checkpoint remains `89e81df0…`. |
| 1 | `v4-phase1/v3-control-causal` | 640 development frames, causal; historical crop definitions | 30.270709 dB, cold 26.267388; zero validation errors. Superseded by complete crop-coverage run below. |
| 1 | `v4-phase1/v3-control-reset16` | 640 development frames, resets every 16; historical crops | 29.607780 dB, 40 resets; zero validation errors. Superseded by complete crop-coverage run below. |
| 1 | `v4-phase1/v3-flip-causal` | Official LDR-FLIP, all 640 frames | Mean 0.10775234; cold 0.16870498, early 0.12633569, warm 0.10374383. |
| 1 | `v4-phase1/v3-ci-causal` | 1,000 sequence resamples, v3 vs its zero-head guide | PSNR +2.1952 dB [1.8758, 2.5867]; FLIP −0.02385 [−0.03108, −0.01717]. Not a v4 comparison. |
| 1 | `v4-phase1/v3-crops-causal` | 24 historical selections, v3 vs zero-head guide | Smooth/edge/texture MSE ratios 0.238/0.773/0.139; no early coverage. Superseded by reference-only extension. |
| 1 | `v4-phase1/flip-smoke-causal` | Official FLIP on separate causal smoke run | Pass; same learned scores as saved-control smoke. |
| 1 | `v4-phase1/bootstrap-two-runs-smoke` | Compare two separately scored evaluation directories | All defined differences and interval endpoints zero; undefined strata null. |
| 1 | `v4-phase1/early-crop-tests` | Owner-approved frame-3 reference-only extension | Ten tests pass; all 24 historical selections preserved, sixteen early selections added with reference hashes. |
| 1 | `v4-phase1/v3-flip-causal-complete` | Official LDR-FLIP, final causal evaluation | Mean 0.10775234; cold/early/warm 0.16870498/0.12633569/0.10374383. |
| 1 | `v4-phase1/v3-ci-causal-complete` | 1,000 paired sequence resamples vs zero-head guide | Overall PSNR +2.1952 dB [1.8758, 2.5867]; all four metrics and all age strata retained. |
| 1 | `v4-phase1/v3-crops-causal-complete` | 40 historical + reference-only early selections | Early smooth/edge/texture MSE ratios 0.2041/0.4874/0.0735 vs guide; cold and warm coverage retained. |
| 1 | `v4-phase1/v3-flip-reset16-complete` | Official LDR-FLIP, final reset-16 evaluation | Mean 0.11928666; cold/early 0.16477328/0.12532265; warm undefined. |
| 1 | `v4-phase1/v3-ci-reset16-complete` | 1,000 paired sequence resamples vs zero-head guide | Overall PSNR +3.2647 dB [2.9263, 3.6505]; all four metrics and all defined age strata retained. |
| 1 | `v4-phase1/v3-crops-reset16-complete` | Same 40 locked selections, reset-aware scoring | Early ratios unchanged; age-15 smooth/edge/texture MSE ratios 0.4217/0.7084/0.1277. |
| 1 | `v4-phase1/verify-control` | Full artifact/coverage/reproduction audit | Pass: 3,840 PNGs and 3,840 float images reproduce exactly; references/protocols/CIs/crops/profile agree; confirmation untouched. |
| 1 | `v4-phase1/final-handoff-checks` | Committed-result identity, fmt, metric/crop/FLIP tests, published evidence | Pass; report is byte-identical to audited output, 19 tests pass, README images and published freeze unchanged. |

After the owner merged PR #23, [CI run 36298665573](https://github.com/kvark/ommatidia/actions/runs/36298665573)
passed all five jobs on `main` at `a153c1e`, including LavaPipe and Linux/macOS/Windows
builds. Phase 0 is complete; upstream publication is an owner-approved, non-blocking
follow-up. The tested local compiler patch remains in use.

Earlier sprint detail is [archived](archive/quality-week.md). The declared v3
control is `89e81df0…`; neither interrupted-run snapshots nor width-32 weights
initialize v4. [Phase 1 measurements](phase1-results.md) establish the control
under both evaluation protocols, with full metrics, intervals and crop coverage.

Phase 2 diagnostic checkpoint: `da4540def5f0d0d40d8c88122c26ecac57b8d406d9cff4221ef9eeaef2b72dab`
at `runs/v4-phase2/single-scene-2000/model/model.safetensors`. It is a one-scene
sanity fit, not the final candidate and not a warm-start source for production.
