# v4 run ledger

Plan: [PLAN.md](../PLAN.md). Paths are relative to the repository. Each new run
uses `scripts/record-run.py`; its manifest, log and inputs remain under `runs/`.
Only development selects models. Confirmation is reserved for Phase 6.

| Phase | Run | Budget / purpose | Outcome |
|---|---|---|---|
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

After the owner merged PR #23, [CI run 36298665573](https://github.com/kvark/ommatidia/actions/runs/36298665573)
passed all five jobs on `main` at `a153c1e`, including LavaPipe and Linux/macOS/Windows
builds. Phase 0 is complete; upstream publication is an owner-approved, non-blocking
follow-up. The tested local compiler patch remains in use.

Earlier sprint detail is [archived](archive/quality-week.md). The declared v3
control is `89e81df0…`; neither interrupted-run snapshots nor width-32 weights
initialize v4. Phase 1 will measure the control under both evaluation protocols.
