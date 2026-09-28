# v4 run ledger

Plan: [PLAN.md](../PLAN.md). Paths are relative to the repository. Each new run
uses `scripts/record-run.py`; its manifest, log and inputs remain under `runs/`.
Only development selects models. Confirmation is reserved for Phase 6.

| Phase | Run | Budget / purpose | Outcome |
|---|---|---|---|
| 5 | `v4-phase5/capacity-queue-recovery` | Finish remaining first-seed runs, both full protocols and final curve audits in a transient user job | Running under `ommatidia-phase5-capacity-20260928.service`; stop on any failure, no selection or second seeds. |
| 5 | `v4-phase5/capacity-retry-recipe-check` | Validate fresh retry directory and unchanged numerical command; dry-run remaining first-seed launch recipes | Pass: retry command differs only in output directory, all three launch recipes pass, unsafe suffix rejected. |
| 5 | `v4-phase5/capacity-w16-l4-interruption` | Read-only audit of the interrupted attempt: process absence, loss prefix and checkpoint availability | 792 finite updates, max loss 0.544915, no spikes >1 or resumable checkpoint; all four execution processes absent. Cause unestablished; originals unchanged. |
| 5 | `v4-phase5/capacity-200-w16-l4-seed1-60000-retry1` | Repeat interrupted 16×4 attempt from scratch, same seed/data/60k schedule, fresh output directory | Recovery launched; no optimizer resume or numerical command change. |
| 5 | `v4-phase5/capacity-w16-l4-seed1-curve-60000` | Complete scheduled causal checkpoint curve and final weights/optimizer/state/sampling audit for 16×4 | Queued after successful training completion; no checkpoint selection. |
| 5 | `v4-phase5/capacity-w32-l3-seed1-curve-60000` | Same complete curve and final-training audit for 32×3 | Queued after successful training completion; no checkpoint selection. |
| 5 | `v4-phase5/capacity-w32-l4-seed1-curve-60000` | Same complete curve and final-training audit for 32×4 | Queued after successful training completion; no checkpoint selection. |
| 5 | `v4-phase5/capacity-40-vs-200-w16-l3-seed1` | Pair the completed 40- and 200-scene runs at identical width/depth, seed and budget, both development protocols | Causal/reset-16 PSNR gains +1.754 [1.175, 2.671]/+1.525 [1.045, 2.305] dB; FLIP improves. Cold smooth ratio 1.230 [0.726, 1.628]: no established crop improvement. |
| 5 | `v4-phase5/capacity-w16-l3-seed1-curve-60000` | Final 200-scene 16×3 curve: paired development intervals, full loss prefix and completed-training audit | 27.7914 dB; +0.057 [−0.141, 0.252] dB versus 50k, −2.479 [−3.209, −1.856] versus v3. Audit passes: 480,000 windows, finite weights/moments/state, exact reload; one loss outlier. |
| 5 | `v4-phase5/capacity-w16-l3-seed1-curve-50000` | Scheduled 200-scene checkpoint: paired development intervals and immutable loss prefix through 50,000 updates | 27.7346 dB; +0.007 [−0.174, 0.195] dB versus 40k, −2.536 [−3.357, −1.865] versus v3. Temporal ratio 0.83966 [0.74242, 0.90620]; one loss outlier through 50k. |
| 5 | `v4-phase5/capacity-w16-l3-seed1-curve-40000` | Scheduled 200-scene checkpoint: paired development intervals and immutable loss prefix through 40,000 updates | 27.7273 dB; +0.647 [0.414, 0.875] dB versus 30k, −2.543 [−3.317, −1.821] versus v3. Temporal ratio 0.87143 [0.76644, 0.94542]; one loss outlier through 40k. |
| 5 | `v4-phase5/capacity-w16-l3-seed1-curve-30000` | Scheduled 200-scene checkpoint: paired development intervals and immutable loss prefix through 30,000 updates | 27.0808 dB; +0.224 [−0.173, 0.644] dB versus 20k, −3.190 [−4.080, −2.404] versus v3. Temporal ratio 0.91559 [0.81387, 0.98831]; one loss outlier through 30k. |
| 5 | `v4-phase5/capacity-w16-l3-seed1-curve-20000` | Scheduled 200-scene checkpoint: paired development intervals and immutable loss prefix through 20,000 updates | 26.8564 dB; +1.335 [0.596, 1.924] dB versus 10k, −3.414 [−4.628, −2.487] versus v3. One finite loss outlier; fixed budget continues. |
| 5 | `v4-phase5/capacity-w16-l3-seed1-curve-10000` | First scheduled 200-scene checkpoint: causal development intervals against v3 and immutable loss prefix | 25.5216 dB; delta −4.749 [−5.397, −4.108] dB, energy 0.94467, temporal ratio 0.97140 [0.85282, 1.05723]. No loss outliers; continue the fixed budget. |
| 5 | `v4-phase5/capacity-comparison-checks` | Verify paired report comparisons against raw-pixel crops, self-pairs and the fixed 10%/CI rule; no new model evaluation | Four checks pass: pixel-exact cached pairing, zero self-differences, mismatch rejection and both threshold conditions. No ranking or selection. |
| 5 | `v4-phase5/capacity-prefix-timing` | Read-only timing of updates 11–2,500 on 40 versus 200 scenes, identical executable/settings; retain immutable CSV prefixes | 14.16 vs 5.38 updates/s: step/wait ~64 ms in both, input wait 0.13 →116.48 ms (62.6% of 200-scene wall time). Observational, not an isolated speed claim; run unchanged. |
| 5 | `v4-phase5/budget-report-audit` | Verify completed reports, exact causal metric reproduction, all shared references and fixed alpha maps/histograms | Ancillary audit failed: Python f64 binning differs by one boundary sample from the writer's f32 multiplication. Reporting unchanged; corrected snapshotted audit follows. |
| 5 | `v4-phase5/budget-report-audit-fixed` | Same report audit with the writer's f32 histogram arithmetic and recorded source | Pass: all 640 causal metric rows reproduce exactly; 1,280 references share the verified controls; all 12 alpha maps/histograms and PNG headers agree. |
| 5 | `v4-phase5/budget-curve-60000` | Final budget curve, whole-sequence intervals and complete checkpoint/optimizer/sampler audit | Pass: finite weights/Adam/state and exact reload, 480,000 windows. Final 26.0374 dB; change vs 50,000 −0.357 [−0.586, −0.136] dB while loss falls. |
| 5 | `v4-phase5/budget-dev-causal` | Final 40-scene-budget checkpoint, full causal development, fixed alpha frames and verified shared references | Complete: 26.0374 dB on all 640 frames; fixed alpha outputs saved and 640 reference files shared. |
| 5 | `v4-phase5/budget-score-causal` | Official FLIP, paired frame/crop intervals and lighting-final comparisons against v3 | Interrupted: launcher exited 143 during FLIP; no child remains live. Partial artifacts and stale running manifest retained; no training failure. |
| 5 | `v4-phase5/budget-score-causal-retry` | Same scoring on the completed immutable causal outputs, fresh directory | Complete: cold smooth ratio 0.940 [0.827, 1.307], warm 4.202 [2.523, 6.029]; FLIP 0.19921 vs v3 0.10775. Not promoted. |
| 5 | `v4-phase5/budget-dev-reset16` | Same final budget checkpoint, development with periodic cuts and fixed alpha frames | Complete: 25.9030 dB on 640 frames/40 cuts; fixed alpha outputs saved and 640 reference files shared. |
| 5 | `v4-phase5/budget-score-reset16` | Same scoring under periodic cuts | Complete: PSNR delta −3.705 [−5.097, −2.776] dB, cold smooth ratio identical; FLIP 0.20092 vs v3 0.11929. Warm strata null. |
| 5 | `v4-phase5/capacity-200-w16-l3-seed1-60000` | 200 scenes, width 16/three levels, seed 1, 60,000 updates from scratch | Complete: 27.7914 dB, 17.50% cold windows, one finite loss outlier, exact reload; all 50 capture/provenance hashes match the admitted corpus. |
| 5 | `v4-phase5/capacity-w16-l3-seed1-dev-causal` | Full causal development and fixed alpha frames for 16×3, seed 1 | Complete: 27.7914 dB on 640 frames; all 640 references verified/shared, fixed alpha maps/histograms saved. |
| 5 | `v4-phase5/capacity-w16-l3-seed1-score-causal` | FLIP, paired frame/crop intervals and lighting-final comparisons | Complete: cold smooth 1.155 [0.687, 1.612], warm 3.625 [2.156, 5.139], FLIP 0.17455 versus v3 0.10775. Spatial/warm/lighting misses retained; no selection. |
| 5 | `v4-phase5/capacity-w16-l3-seed1-dev-reset16` | Same checkpoint/development with periodic cuts and fixed alpha frames | Complete: 27.4277 dB on 640 frames/40 cuts; all references verified/shared, fixed alpha maps/histograms saved. |
| 5 | `v4-phase5/capacity-w16-l3-seed1-score-reset16` | Same scoring under periodic cuts | Complete: PSNR delta −2.180 [−2.837, −1.662] dB, FLIP 0.17896 versus v3 0.11929; temporal ratio 0.69502 [0.62306, 0.75750]. Warm strata null. |
| 5 | `v4-phase5/capacity-200-w16-l4-seed1-60000` | 200 scenes, width 16/four levels, seed 1, 60,000 updates from scratch | Interrupted after 792 recorded updates; trainer/recorder/queue absent, no optimizer checkpoint. Original stale manifest/artifacts preserved; cause unestablished. |
| 5 | `v4-phase5/capacity-w16-l4-seed1-dev-causal` | Full causal development and fixed alpha frames for 16×4, seed 1 | Queued after training. |
| 5 | `v4-phase5/capacity-w16-l4-seed1-score-causal` | FLIP, paired frame/crop intervals and lighting-final comparisons | Queued after causal evaluation. |
| 5 | `v4-phase5/capacity-w16-l4-seed1-dev-reset16` | Same checkpoint/development with periodic cuts and fixed alpha frames | Queued after training. |
| 5 | `v4-phase5/capacity-w16-l4-seed1-score-reset16` | Same scoring under periodic cuts | Queued after reset-16 evaluation. |
| 5 | `v4-phase5/capacity-200-w32-l3-seed1-60000` | 200 scenes, width 32/three levels, seed 1, 60,000 updates from scratch | Queued after 16×4 reporting. |
| 5 | `v4-phase5/capacity-w32-l3-seed1-dev-causal` | Full causal development and fixed alpha frames for 32×3, seed 1 | Queued after training. |
| 5 | `v4-phase5/capacity-w32-l3-seed1-score-causal` | FLIP, paired frame/crop intervals and lighting-final comparisons | Queued after causal evaluation. |
| 5 | `v4-phase5/capacity-w32-l3-seed1-dev-reset16` | Same checkpoint/development with periodic cuts and fixed alpha frames | Queued after training. |
| 5 | `v4-phase5/capacity-w32-l3-seed1-score-reset16` | Same scoring under periodic cuts | Queued after reset-16 evaluation. |
| 5 | `v4-phase5/capacity-200-w32-l4-seed1-60000` | 200 scenes, width 32/four levels, seed 1, 60,000 updates from scratch | Queued after 32×3 reporting. |
| 5 | `v4-phase5/capacity-w32-l4-seed1-dev-causal` | Full causal development and fixed alpha frames for 32×4, seed 1 | Queued after training. |
| 5 | `v4-phase5/capacity-w32-l4-seed1-score-causal` | FLIP, paired frame/crop intervals and lighting-final comparisons | Queued after causal evaluation. |
| 5 | `v4-phase5/capacity-w32-l4-seed1-dev-reset16` | Same checkpoint/development with periodic cuts and fixed alpha frames | Queued after training. |
| 5 | `v4-phase5/capacity-w32-l4-seed1-score-reset16` | Same scoring under periodic cuts | Queued after reset-16 evaluation. |
| 5 | `v4-phase5/capacity-recipe-check` | Dry-run all four planned 200-scene configurations and their fixed reporting commands; no training | Pass: 50 ordered captures/200 unique scenes, matching executable, prior per-size GPU gates and fixed from-scratch schedule; training/report commands prepared without executing them. |
| 5 | `v4-phase5/budget-curve-50000` | Scheduled 50,000-update checkpoint: paired sequence intervals and immutable loss prefix | 26.3941 dB; change vs 40,000 +0.026 [−0.335, +0.277] dB. Training loss is ~26% below 30,000 but development remains flat; one finite outlier. |
| 5 | `v4-phase5/candidate-recipe-check` | Dry-run the fixed two-protocol evaluation/scoring recipe against the completed sanity bundle; no repeated inference | Pass: checks completed step, checkpoint/executable identity and finite reload, then prepares four independently recorded commands; no model or protocol change. |
| 5 | `v4-phase5/budget-curve-40000` | Scheduled 40,000-update checkpoint: paired sequence intervals and immutable loss prefix | 26.3681 dB; change vs 30,000 −0.019 [−0.414, +0.356] dB while mean training loss falls ~21%. Energy 1.00538; continue the fixed budget. |
| 5 | `v4-phase5/budget-curve-30000` | Half-budget checkpoint: complete causal development curve, paired sequence intervals and frozen loss prefix | 26.3870 dB; change vs 20,000 +0.340 [−0.160, +0.888] dB. Energy 1.03080, temporal ratio 0.85051; one finite outlier. |
| 5 | `v4-phase5/budget-curve-20000` | Second scheduled checkpoint: paired sequence intervals against 10,000 updates and v3, with the complete loss prefix | 26.0469 dB; improvement vs 10,000 is +0.562 [−0.016, +1.201] dB. Energy 0.97125; one finite outlier. Continue the fixed budget. |
| 5 | `v4-phase5/budget-outlier-15991` | Read-only observation check for the first long-budget finite loss outlier; preserve recorded cursor diagnostics | Batch loss 1.0605, one cold HDR cursor 8.4269; predicted peak 1,225.6 vs target 120.7. Motion ≤0.521 pixels, not the earlier large-motion signature; no training changes. |
| 5 | `v4-phase5/budget-curve-10000` | First scheduled budget-curve checkpoint: metrics-only causal development, whole-sequence intervals and frozen loss prefix | 25.4850 dB; delta vs v3 −4.786 [−5.923, −3.779] dB. No loss spikes >1; continue the unchanged 60,000-step budget. |
| 5 | `v4-phase5/ladder-checks` | Workspace regular tests/fmt plus frame/crop reporting tests and unchanged-gallery verification after capacity-test parameterization | 99 regular Rust tests, seven frame and twelve crop tests pass; fmt and gallery verification pass. |
| 5 | `v4-phase5/budget-40-w16-l3-seed1-60000` | Budget curve: original 40 scenes, width 16/three levels, seed 1, 60,000 updates from scratch; causal development every 10,000 | Complete: 26.0374 dB, 17.50% cold windows, one finite loss outlier and exact reload. Overlapped checks/reporting exclude isolated throughput claims. |
| 5 | `v4-phase5/capacity-16x4-radv` | Debug RADV: recurrent HDR/reset parity, every-parameter f64 gradients and production-extent gradient directions | Three tests pass, zero validation errors. |
| 5 | `v4-phase5/capacity-32x3-radv` | Width 32, three levels: same three RADV correctness gates | Three tests pass, zero validation errors. |
| 5 | `v4-phase5/capacity-32x4-radv` | Width 32, four levels: same three RADV correctness gates | Three tests pass, zero validation errors. |
| 5 | `v4-phase5/corrected-noise-causal` | Final corrected sanity checkpoint on independent input noise, causal; gate ≥32.6 dB | Pass: 33.0874 dB, +0.5999 dB versus initial fit. |
| 5 | `v4-phase5/corrected-noise-reset1` | Same independent-noise stream, resetting history every frame; gate ≥31.6 dB | Pass: 31.8499 dB, +0.4877 dB versus initial fit. |
| 5 | `v4-phase5/corrected-dev-causal` | Corrected sanity checkpoint, full ten-scene development with fixed alpha frames and shared verified references | 20.2038 dB on 640 frames; alpha maps/histograms saved, all 640 reference files shared. Single-scene fit, not promoted. |
| 5 | `v4-phase5/corrected-dev-reset16` | Same checkpoint/development with periodic cuts | 20.0869 dB on 640 frames/40 cuts; fixed alpha maps/histograms saved, all 640 reference files shared. |
| 5 | `v4-phase5/corrected-score-causal` | Official FLIP, paired frame/crop intervals, and fixed lighting-final comparisons against v3 | Complete: PSNR delta −10.067 [−12.060, −8.491] dB, cold smooth ratio 3.866 [1.958, 6.016]; not promoted. |
| 5 | `v4-phase5/corrected-score-reset16` | Same scoring for periodic cuts | Complete: PSNR delta −9.521 [−11.440, −7.958] dB; cold smooth ratio identical, unsupported warm strata null. |
| 5 | `v4-phase5/capacity-test-build` | Parameterize existing recurrence/reference-gradient gates by the planned width/depth, without changing the model | Format, workspace test build and all-target Clippy pass. |
| 5 | `v4-phase5/capacity-16x4-lavapipe` | Width 16, four levels: debug recurrent HDR/reset parity and every-parameter f64 gradient gates, fused/unfused | Two tests pass, zero validation errors. Background CPU checks exclude sanity throughput from speed claims. |
| 5 | `v4-phase5/capacity-32x3-lavapipe` | Width 32, three levels: recurrence, f64 gradients and 128² production-extent gradient directions | Three tests pass, zero validation errors. |
| 5 | `v4-phase5/capacity-32x4-lavapipe` | Width 32, four levels: same three correctness gates | Three tests pass, zero validation errors. |
| 5 | `v4-phase5/capacity-16x4-extent-lavapipe` | Width 16, four levels: 128² production-extent gradient directions | Pass, zero validation errors. |
| 5 | `v4-phase5/final-frame-bootstrap-tests` | Lighting-final reporting: fixed sequence selection, individual regression visibility and sequence-cluster intervals | Seven frame and twelve crop tests pass; published-gallery verification unchanged. |
| 5 | `v4-phase5/final-frame-bootstrap-smoke` | Existing sanity/v3 scores, both protocols: exercise fixed lighting-final reporting without rerunning models | Pass: selects exactly frames 6:63/7:63; all existing bucket/interval outputs remain identical. |
| 5 | `v4-phase5/sanity-corrected-10000` | Fresh seed-1 single-scene fit with both verified corrections; unchanged data, 10,000-step schedule and thresholds | Complete: no loss spikes >1, finite/exact reload, 17.72% cold windows; both independent-noise gates pass. |
| 5 | `v4-phase5/replay-5000-audit` | Audit complete corrected seed-7 replay, including parent prefix, optimizer continuation and full development | Ancillary audit assertion failed: counted Adam moments as model parameters. Training completed successfully; corrected audit below. |
| 5 | `v4-phase5/replay-5000-audit-fixed` | Same audit with parameter/optimizer tensor counts distinguished | Pass: 5,000 updates, identical sampling/schedule, no spikes >1 (old: 14); finite weights/moments, exact reload. |
| 5 | `v4-phase5/alpha-evaluation-smoke` | End-to-end CLI alpha outputs: exact raw probabilities/histograms, cold zeros and grayscale PNG headers, plus resume/control smoke | Pass on debug LavaPipe, including exact resume/reload; zero validation errors. |
| 5 | `v4-phase5/reference-sharing-checks` | Byte-verified immutable control references: hard links/copy fallback, mismatch and overwrite rejection | 99 regular tests, Clippy/fmt/debug build and unchanged gallery verification pass. |
| 5 | `v4-phase5/reference-sharing-smoke` | Full debug evaluation/resume smoke plus byte identity and actual shared-file count | Pass on LavaPipe: exact resume/reload, both reference frames shared, zero validation errors. |
| 5 | `v4-phase5/spike-pair-audit` | Compare complete seed-7 prefixes, data/settings/sampler identity, outlier counts and loss ranges | Identical 4,000 windows/settings/data; update-303 loss 557,953.06 → 0.02509; corrected maximum 0.09408, no spikes >1. |
| 5 | `v4-phase5/reset-motion-regression-before` | Counterfactual regression: change only irrelevant cold-frame motion to the observed spike magnitude | Fails as expected: cold feature 3584 changes from 0 to 1,305 despite absent history. |
| 5 | `v4-phase5/reset-motion-gpu-before` | Same cold-motion counterfactual through the unfixed native GPU path and recurrent state | Invalid check: used an older workspace test artifact and matched zero tests. Corrected below. |
| 5 | `v4-phase5/reset-motion-gpu-before-corrected` | Run the newly built regression artifact on debug LavaPipe | Fails as expected: cold output changes from 1.41825 to 0.79996 solely from absent-frame motion. |
| 5 | `v4-phase5/reset-motion-checks` | Zero only cold motion features; full fmt/tests/Clippy/build and schema-3 resume fixtures | 99 regular tests, all-target Clippy, fmt and release build pass. |
| 5 | `v4-phase5/reset-motion-lavapipe` | All GPU checks with cold-motion independence and schema-3 resume, debug LavaPipe | Thirteen tests pass, zero validation errors; cold output/state now bit-identical under motion-only perturbations. |
| 5 | `v4-phase5/reset-motion-evaluation-smoke` | Debug CLI: periodic evaluation, interrupted resume, reload, saved control, and reset-1/16 | Pass: resumed/reloaded CSV byte-identical, zero validation errors. |
| 5 | `v4-phase5/reset-motion-debug-cli` | Rebuild standalone debug trainer for the exact CLI evaluation/resume smoke | Pass. |
| 5 | `v4-phase5/image-edge-noise-causal` | Border-only checkpoint, independent noise, archived matching runtime, causal | 33.1141 dB: passes ≥32.6 dB, +0.6266 dB versus initial fit. |
| 5 | `v4-phase5/image-edge-noise-reset1` | Same border-only checkpoint/runtime, history reset every frame | 31.7441 dB: passes ≥31.6 dB, +0.3820 dB versus initial fit. |
| 5 | `v4-phase5/reset-motion-radv` | Full GPU checks with reset-motion correction and schema-3 resume, debug RADV | Thirteen tests pass, zero validation errors. |
| 5 | `v4-phase5/image-edge-spike-500` | Controlled seed-7 40-scene prefix, corrected borders but original reset-motion features; archived runtime | Completes; update 303 spikes to 557,953.06 despite border fix. |
| 5 | `v4-phase5/reset-motion-spike-500` | Identical prefix with only cold-motion feature correction added | Completes with no spikes >1 (maximum 0.09408); finite parameters and exact reload. |
| 5 | `v4-phase5/stable-runtime-build` | Locked release trainer after numerical fixes and storage-only reference sharing | Pass; numerical model/loop unchanged by reference storage. |
| 5 | `v4-phase5/reset-motion-replay-5000` | Finish the original seed-7 5,000-step diagnostic via true schema-3 resume from step 500, then full causal development | Complete: causal 24.8592 dB (old 24.6385), cold 23.9078 (old 21.1600); no loss spikes >1. Not a production candidate. |
| 5 | `v4-phase5/archive-image-edge-runtime` | Preserve the executing border-only trainer/evaluator before testing reset-motion changes | Pass: archived hash exactly matches the running fit's recorded executable. |
| 5 | `v4-phase5/image-edge-checks` | Corrected crop-loss masks: normalization, real-edge supervision, gradients/restore fixtures; fmt/tests/Clippy/build | Compile failed on an inferred integer bitmask type; corrected explicit usize. |
| 5 | `v4-phase5/image-edge-checks-fixed` | Repeat after bitmask type fix; include actual GPU mask transfer assertions | 98 regular tests, fmt/Clippy and release build pass; GPU suites follow. |
| 5 | `v4-phase5/image-edge-radv` | Corrected loss masks: all GPU gates, f64 gradients, packing/state carry and schema-2 resume, debug RADV | Twelve tests pass, zero validation errors. |
| 5 | `v4-phase5/image-edge-lavapipe` | Same full GPU checks on debug LavaPipe | Twelve tests pass, zero validation errors. |
| 5 | `v4-phase5/spike-observations` | Inspect captured geometry/motion at the two replayed cold prediction spikes | First-frame motion peaks at 135 and 1,305 HR pixels, then drops below 1; candidate cause for a controlled cold-input test, not yet a causal attribution. |
| 5 | `v4-phase5/sanity-image-edges-10000` | Repeat seed-1 sanity from scratch with only corrected image-edge loss masks; unchanged data/schedule/budget | Complete: finite/exact reload, 17.72% cold windows; both independent-noise gates pass. Background CPU checks exclude this run from speed comparisons. |
| 5 | `v4-phase5/sanity-build` | Locked release trainer at the merged caller-encoder Meganeura pin; unchanged model/loop | Pass; previous debug correctness gates cover the identical merged source tree. |
| 5 | `v4-phase5/sanity-10000` | From scratch, one scene, seed 1, 10,000 updates; B=8, k=4, 64² LR crops | Complete: 621.90 training seconds, 16.08 updates/s, 17.72% cold windows; independent-noise gates follow. |
| 5 | `v4-phase5/diagnostic-radv` | Full reconstruction and cursor GPU gates after adding read-only diagnostics, debug RADV | Twelve tests pass, zero validation errors; alpha and diagnostic readbacks preserve outputs/resume exactly. |
| 5 | `v4-phase5/absolute-bootstrap-tests` | Absolute energy and paired ratio confidence intervals, including zero denominators | Six evaluation and twelve crop tests pass; published-gallery verification unchanged. |
| 5 | `v4-phase5/sanity-noise-causal` | Final sanity checkpoint, independent input noise, causal evaluation | 32.4875 dB; misses ≥32.6 dB sanity gate. Longer ladder held for model/loop diagnosis. |
| 5 | `v4-phase5/sanity-noise-reset1` | Same checkpoint/noise stream, history reset every frame | 31.3621 dB; misses ≥31.6 dB sanity gate. No threshold/data change. |
| 5 | `v4-phase5/spike-replay-500` | Phase 3 seed-7 ordered 40-scene replay, original 5,000-step schedule, stop after 500; read-only outlier diagnostics | Three spikes reproduce (303/396/397): cold-window prediction explosion, not extreme targets; no model promoted. |
| 5 | `v4-phase5/diagnostic-lavapipe-full` | Full reconstruction and cursor GPU gates with diagnostic changes, debug LavaPipe | Twelve tests pass, zero validation errors. |
| 5 | `v4-phase5/sanity-noise-linear` | Exact causal sanity rerun retaining floats for full-frame versus interior-error diagnosis | Pass: causal scores byte-identical; all 64 references match the archived Phase 2 diagnostic. |
| 5 | `v4-phase5/sanity-dev-causal` | Initial sanity checkpoint, complete development protocol, locked crops and three fixed alpha frames | 19.8347 dB across 640 frames; alpha maps/histograms saved. Single-scene fit, not promoted. |
| 5 | `v4-phase5/sanity-dev-reset16` | Same checkpoint/development, cuts every 16 frames; locked crops and alpha frames | 19.8179 dB across 640 frames/40 cuts; alpha maps/histograms saved. |
| 5 | `v4-phase5/sanity-score-causal` | Official FLIP, paired frame/crop confidence intervals against learned v3 control | Complete: PSNR delta −10.436 dB [−12.251, −9.030], cold smooth ratio 3.080 [1.574, 4.245]; no promotion. |
| 5 | `v4-phase5/sanity-score-reset16` | Same scoring for periodic cuts | Complete; 640 frames, frame/crop intervals retained, unsupported warm strata null. |
| 5 | `v4-phase5/sanity-error-localization` | Compare full-frame/interior errors with the archived Phase 2 diagnostic; unchanged references | Outer 4-pixel border accounts for 28.15% of error (Phase 2: 6.85%); 4-pixel-trimmed PSNR 33.65 dB is diagnostic only, not a gate substitution. |
| 5 | `v4-phase5/diagnostic-checks` | Format, workspace tests/Clippy, diagnostic-output parity fixtures and release build | Pass: 96 regular tests, Clippy and release build. GPU diagnostics checked separately. |
| 5 | `v4-phase5/archive-sanity-runtime` | Preserve the executing pre-instrumentation trainer for controlled replay | Pass; archived executable hash exactly matches the sanity run's input. |
| 5 | `v4-phase5/diagnostic-lavapipe` | Alpha-output parity plus cursor/diagnostic readback and exact resume, debug LavaPipe | Three tests pass, zero validation errors; alpha output leaves reconstruction bit-identical. |
| 5 | `v4-phase5/crop-bootstrap-tests` | Paired whole-sequence crop intervals with area weighting and missing strata | Pass: twelve crop tests, five frame/bootstrap tests, unchanged published-gallery verification. |
| Integration | `blade-encoder/format` | Format caller-encoder example, API documentation and parity test | Pass. |
| Integration | `blade-encoder/build` | Build all workspace targets with Meganeura `7c29497`; debug assertions, debug symbols disabled | Pass. |
| Integration | `blade-encoder/radv-gpu` | Full reconstruction GPU suite, including caller-encoder recurrence/cuts; debug RADV | Failed validation: new test reused caller command buffers without frame fences; corrected test lifetime handling. |
| Integration | `blade-encoder/lavapipe-gpu` | Same reconstruction GPU suite on debug LavaPipe | Failed validation and SIGSEGV from the same test command-buffer reuse error. Superseded by fenced run. |
| Integration | `blade-encoder/checks` | Workspace regular tests, all-target Clippy, formatting and published-gallery verification | Pass: 94 regular tests, Clippy/fmt/gallery. GPU checks recorded separately. |
| Integration | `blade-encoder/fenced-build` | Format and rebuild corrected caller-encoder parity test and standalone example | Pass. |
| Integration | `blade-encoder/radv-fenced` | Nine reconstruction GPU gates, two cursor/resume gates and standalone example; debug RADV | Pass: eleven gates and example, zero validation errors; caller-encoder output is bit-exact across recurrence/cuts. |
| Integration | `blade-encoder/lavapipe-fenced` | Same eleven GPU gates and standalone example; debug LavaPipe | Pass: eleven gates and example, zero validation errors; caller-encoder output is bit-exact across recurrence/cuts. |
| Integration | `blade-encoder/final-checks` | Final all-target Clippy, workspace tests, formatting and gallery check after fence correction | Pass: 94 regular tests, Clippy/fmt/gallery; model, training data and published quality results unchanged. |
| 4 | `v4-phase4/final-checks` | Workspace fmt, release all-target Clippy/tests, corpus fixtures and published-gallery verifier | Pass: 94 regular Rust tests, nine corpus tests; fmt/Clippy/gallery pass. Unchanged GPU gates were not rerun. |
| 4 | `v4-phase4/completion-audit` | Requirement-by-requirement cross-check of command identity, preflight chronology, hashes, counts and run coverage | Pass: all 40 batches, 200 scenes, 2,560 visibility samples and 9,000 loader probes agree; all 61 runs have ledger entries, including the superseded ancillary audit failure. |
| 4 | `v4-phase4/restore-admission-build` | Restore standalone admission executable after workspace-test feature unification | Pass; byte-identical to the executable used by the successful full loader check. |
| 4 | `v4-phase4/loader-check` | Actual Phase 3 loader on the checked-in 200-scene manifest, larger than RAM | Pass: all 50 file/provenance hashes, all reference means, 9,000 crop/gain decodes; zero model evaluations or optimizer updates. |
| 4 | `v4-phase4/final-loader-build` | Rebuild the CPU-only admission utility after rebasing onto the owner's Phase 3 merge | Pass; source tree identical across the rebase, Rust source snapshotted explicitly. |
| 4 | `v4-phase4/verify-run` | Exhaustive admission of all 200 scenes: hashes, every record, scene membership and complete catalog trajectories | Pass: 12,800 finite records, valid references; all 2,560 catalog frames visible (minimum 1.534%). Original/protected inputs unchanged; 52,596,575,360 bytes total. |
| 4 | `v4-phase4/preflight-coverage-all-families` | Audit exact required seed groups, chronology and full held-out catalog | Pass: all training/dev/audit/confirmation/diagnostic/ancestry records covered before capture; 22 training families disjoint from all 12 held-out families (8 ABO, 4 unused legacy interiors). |
| 4 | `v4-phase4/preflight-coverage` | Independently audit required preflight coverage | Required seed groups/chronology pass; ancillary count assertion fails because the root catalog includes unused legacy interiors as well as 8 held-out ABO families. No data/split failure. |
| 4 | `v4-phase4/visibility-roundtrip-tests` | Compare coverage sidecars by exact float32 bits despite different JSON decimal encodings | Nine tests pass; one-ULP differences still fail. No capture or threshold change. |
| 4 | `v4-phase4/final-script-tests` | Final arrangement of all Phase 4 verifier fixtures | Eight pass, including header corruption, late-frame visibility failures and numerical admission rejection. |
| 4 | `v4-phase4/loader-smoke` | Actual mapped-loader admission of the first four-scene capture | Pass: matching hashes, all reference means valid, 180 cold/middle/final crop+gain decodes; no model evaluation or optimizer update. |
| 4 | `v4-phase4/numerics-tests` | Extend data-verifier tests to truncated files, NaN/Inf, negative and black references | Eight tests pass in the pinned evaluation environment; CI includes them. |
| 4 | `v4-phase4/data-smoke` | Full numerical scan of the first new static capture | Pass: 256 finite records, nonnegative/nonblack references; peak reference radiance 40.6875. |
| 4 | `v4-phase4/script-tests-final` | Add fixed-header size/contract rejection | Seven tests pass; subsequently extended with numerical admission fixtures. |
| 4 | `v4-phase4/format` | Workspace formatting with CPU-only loader admission utility | Pass. |
| 4 | `v4-phase4/loader-build` | Build CPU-only `check-corpus` against the Phase 3 loader | Pass; no model architecture or training-loop changes. |
| 4 | `v4-phase4/loader-clippy` | Release Clippy for the admission utility, warnings denied | Pass. |
| 4 | `v4-phase4/capture-static-00` | Four fresh static scenes × 64 frames | Complete, 119.8 s wall; release capture, 0 logged validation errors. |
| 4 | `v4-phase4/capture-camera-00` | Four fresh camera scenes × 64 frames | Complete, 119.2 s wall; release capture, 0 logged validation errors. |
| 4 | `v4-phase4/capture-objects-00` | Four fresh objects scenes × 64 frames | Complete, 120.9 s wall; release capture, 0 logged validation errors. |
| 4 | `v4-phase4/capture-lights-00` | Four fresh lights scenes × 64 frames | Complete, 128.3 s wall; release capture, 0 logged validation errors. |
| 4 | `v4-phase4/capture-catalog-00` | Four fresh catalog scenes × 64 frames | Complete, 178.2 s wall; release capture, 0 logged validation errors. All 256 visibility samples pass, minimum 1.927%. |
| 4 | `v4-phase4/capture-static-01` | Four fresh static scenes × 64 frames | Complete, 98.6 s wall; release capture, 0 logged validation errors. |
| 4 | `v4-phase4/capture-camera-01` | Four fresh camera scenes × 64 frames | Complete, 115.1 s wall; release capture, 0 logged validation errors. |
| 4 | `v4-phase4/capture-objects-01` | Four fresh objects scenes × 64 frames | Complete, 118.7 s wall; release capture, 0 logged validation errors. |
| 4 | `v4-phase4/capture-lights-01` | Four fresh lights scenes × 64 frames | Complete, 115.1 s wall; release capture, 0 logged validation errors. |
| 4 | `v4-phase4/capture-catalog-01` | Four fresh catalog scenes × 64 frames | Complete, 180.8 s wall; release capture, 0 logged validation errors. All 256 visibility samples pass, minimum 10.681%. |
| 4 | `v4-phase4/capture-static-02` | Four fresh static scenes × 64 frames | Complete, 113.1 s wall; release capture, 0 logged validation errors. |
| 4 | `v4-phase4/capture-camera-02` | Four fresh camera scenes × 64 frames | Complete, 120.7 s wall; release capture, 0 logged validation errors. |
| 4 | `v4-phase4/capture-objects-02` | Four fresh objects scenes × 64 frames | Complete, 124.2 s wall; release capture, 0 logged validation errors. |
| 4 | `v4-phase4/capture-lights-02` | Four fresh lights scenes × 64 frames | Complete, 120.5 s wall; release capture, 0 logged validation errors. |
| 4 | `v4-phase4/capture-catalog-02` | Four fresh catalog scenes × 64 frames | Complete, 176.2 s wall; release capture, 0 logged validation errors. All 256 visibility samples pass, minimum 9.084%. |
| 4 | `v4-phase4/capture-static-03` | Four fresh static scenes × 64 frames | Complete, 115.4 s wall; release capture, 0 logged validation errors. |
| 4 | `v4-phase4/capture-camera-03` | Four fresh camera scenes × 64 frames | Complete, 98.8 s wall; release capture, 0 logged validation errors. |
| 4 | `v4-phase4/capture-objects-03` | Four fresh objects scenes × 64 frames | Complete, 105.0 s wall; release capture, 0 logged validation errors. |
| 4 | `v4-phase4/capture-lights-03` | Four fresh lights scenes × 64 frames | Complete, 122.0 s wall; release capture, 0 logged validation errors. |
| 4 | `v4-phase4/capture-catalog-03` | Four fresh catalog scenes × 64 frames | Complete, 166.6 s wall; release capture, 0 logged validation errors. All 256 visibility samples pass, minimum 5.498%. |
| 4 | `v4-phase4/capture-static-04` | Four fresh static scenes × 64 frames | Complete, 110.2 s wall; release capture, 0 logged validation errors. |
| 4 | `v4-phase4/capture-camera-04` | Four fresh camera scenes × 64 frames | Complete, 114.5 s wall; release capture, 0 logged validation errors. |
| 4 | `v4-phase4/capture-objects-04` | Four fresh objects scenes × 64 frames | Complete, 120.7 s wall; release capture, 0 logged validation errors. |
| 4 | `v4-phase4/capture-lights-04` | Four fresh lights scenes × 64 frames | Complete, 111.7 s wall; release capture, 0 logged validation errors. |
| 4 | `v4-phase4/capture-catalog-04` | Four fresh catalog scenes × 64 frames | Complete, 188.2 s wall; release capture, 0 logged validation errors. All 256 visibility samples pass, minimum 1.534%. |
| 4 | `v4-phase4/capture-static-05` | Four fresh static scenes × 64 frames | Complete, 104.7 s wall; release capture, 0 logged validation errors. |
| 4 | `v4-phase4/capture-camera-05` | Four fresh camera scenes × 64 frames | Complete, 103.9 s wall; release capture, 0 logged validation errors. |
| 4 | `v4-phase4/capture-objects-05` | Four fresh objects scenes × 64 frames | Complete, 100.4 s wall; release capture, 0 logged validation errors. |
| 4 | `v4-phase4/capture-lights-05` | Four fresh lights scenes × 64 frames | Complete, 128.7 s wall; release capture, 0 logged validation errors. |
| 4 | `v4-phase4/capture-catalog-05` | Four fresh catalog scenes × 64 frames | Complete, 183.8 s wall; release capture, 0 logged validation errors. All 256 visibility samples pass, minimum 1.663%. |
| 4 | `v4-phase4/capture-static-06` | Four fresh static scenes × 64 frames | Complete, 110.2 s wall; release capture, 0 logged validation errors. |
| 4 | `v4-phase4/capture-camera-06` | Four fresh camera scenes × 64 frames | Complete, 114.7 s wall; release capture, 0 logged validation errors. |
| 4 | `v4-phase4/capture-objects-06` | Four fresh objects scenes × 64 frames | Complete, 104.9 s wall; release capture, 0 logged validation errors. |
| 4 | `v4-phase4/capture-lights-06` | Four fresh lights scenes × 64 frames | Complete, 111.4 s wall; release capture, 0 logged validation errors. |
| 4 | `v4-phase4/capture-catalog-06` | Four fresh catalog scenes × 64 frames | Complete, 176.3 s wall; release capture, 0 logged validation errors. All 256 visibility samples pass, minimum 4.239%. |
| 4 | `v4-phase4/capture-static-07` | Four fresh static scenes × 64 frames | Complete, 99.0 s wall; release capture, 0 logged validation errors. |
| 4 | `v4-phase4/capture-camera-07` | Four fresh camera scenes × 64 frames | Complete, 112.4 s wall; release capture, 0 logged validation errors. |
| 4 | `v4-phase4/capture-objects-07` | Four fresh objects scenes × 64 frames | Complete, 127.9 s wall; release capture, 0 logged validation errors. |
| 4 | `v4-phase4/capture-lights-07` | Four fresh lights scenes × 64 frames | Complete, 126.6 s wall; release capture, 0 logged validation errors. |
| 4 | `v4-phase4/capture-catalog-07` | Four fresh catalog scenes × 64 frames | Complete, 173.3 s wall; release capture, 0 logged validation errors. All 256 visibility samples pass, minimum 5.344%. |
| 4 | `v4-phase4/script-tests` | Seed derivation, nested ancestry, command preservation, visibility, no-overwrite and hash rejection | Six tests pass. |
| 4 | `v4-phase4/preflight-run` | Before capture: seed/family splits, protected hashes, archived executable and space | Pass: 160 new seeds disjoint from 305 prior seeds across 943 provenance sources; exact archive binary; 42,077,260,288 capture bytes planned. |
| 4 | `v4-phase4/capture-run` | Sequential orchestration of 40 recorded four-scene capture batches | Complete: 160 new scenes / 10,240 frames in 84.89 min wall; all 40 batches succeed, no training/evaluation. |
| 4 | `v4-phase4/clean-debug` | Make room for the planned 160-scene capture | Removed 49.8 GiB of regenerable debug build artifacts only; 59 GiB free, release runtimes/data/evidence preserved. |
| 3 | `v4-phase3/final-gallery-check` | README/gallery consistency after throughput/docs update | Pass; historical images, checkpoint and frozen selections unchanged. |
| 3 | `v4-phase3/causal-comparison` | Reference-hashed paired comparison against frozen v3; 1,000 sequence resamples | PSNR −5.632 dB [−6.523, −4.935]; FLIP +0.124952 [+0.106281, +0.144641]. This smoke checkpoint is not promoted; temporal difference inconclusive. |
| 3 | `v4-phase3/flip-causal` | Official FLIP on independently reloaded 640-frame development output | Mean 0.232705, cold 0.304717, warm 0.229213; all float outputs finite/nonnegative. Not promoted. |
| 3 | `v4-phase3/verify-run` | Audit all 5,000 updates, 84 parameter/Adam tensors, cursor state, exact reload and protected hashes | Pass; optimizer step 5,000, 40,000 windows, 6,991 cold; gallery/confirmation/crop selections unchanged. |
| 3 | `v4-phase3/train-5000-retry` | From scratch, 40 scenes, 5,000 steps; 640 development frames | Completes in 311.45 timed training seconds, 17.48% cold. Development 24.638 dB, cold 21.160; 14 finite loss spikes >1 (max 547,544.7). Finite/exact reload, zero validation errors; not promoted. |
| 3 | `v4-phase3/reload-5000` | Independent full-development reload with raw float outputs | All 640 frame scores and diagnostics reproduce byte-for-byte; zero validation errors. |
| 3 | `v4-phase3/ci-eight-reload` | Independent process reload of exact CI checkpoint | CSV byte-identical; zero validation errors. |
| 3 | `v4-phase3/ci-eight-updates` | Exact eight-update CI configuration, debug LavaPipe | Pass, finite parameters and bit-exact checkpoint reload; zero validation errors. |
| 3 | `v4-phase3/python-checks-direct` | Evaluator/crop/catalog/render tests and gallery verifier | 21 pass; published evidence and protected selections unchanged. |
| 3 | `v4-phase3/final-unit-space` | Final workspace tests after package-scoped cache cleanup | 94 pass; ten GPU gates run separately. |
| 3 | `v4-phase3/final-clippy-space` | Final workspace/all-target lint | Pass, warnings denied. |
| 3 | `v4-phase3/final-format` | Rust formatting | Pass. |
| 3 | `v4-phase3/python-checks` | Existing evaluator/crop/gallery tests | Discovery skipped the hyphenated filenames and found zero tests; rerun their direct unittest entry points below. |
| 3 | `v4-phase3/train-5000` | First 5,000-step attempt, from scratch on 40 scenes | Stopped around step 240 when concurrent test linking exhausted disk; finite loss, no checkpoint yet. Restarted from scratch after clearing package-scoped debug outputs. |
| 3 | `v4-phase3/final-unit` | Workspace unit suite | Linking exhausted disk; recorder finalization also failed. Cleared regenerable trainer debug outputs before rerun. |
| 3 | `v4-phase3/final-clippy` | Workspace/all-target lint | Disk full while writing build fingerprint; recorder finalization also failed. |
| 3 | `v4-phase3/evaluation-smoke` | Debug LavaPipe: periodic/final evaluation, CLI interrupted resume, reload, control/cuts | Pass; resumed/reloaded CSV byte-identical; zero validation errors. |
| 3 | `v4-phase3/profile-200` | 200 updates, 40 scenes, B=8, k=4, 64² LR crops | 16.074 updates/s, 7.407 M valid pixel-gradients/s, **24.769×** Phase 1; 16.31% cold windows. Finite/exact checkpoint reload; not a candidate. |
| 3 | `v4-phase3/lavapipe-model` | Eight debug GPU model gates, including all-parameter mean accumulation | All pass; zero validation errors. |
| 3 | `v4-phase3/current-release-build` | Current release trainer | Pass. |
| 3 | `v4-phase3/radv-packing` | Independent GPU target/material packing, two 24-frame cursor streams | Pass, target buffers match CPU packing exactly; zero validation errors. |
| 3 | `v4-phase3/radv-accumulation` | Two-frame mean-gradient accumulation, identical and different micro-batches | Pass for every parameter; zero validation errors. |
| 3 | `v4-phase3/clippy-fixed` | Workspace/all-target lint | Pass, warnings denied. |
| 3 | `v4-phase3/radv-model` | Model GPU suite plus accumulation gate | Seven existing checks pass; the new single-frame accumulation fixture cannot differentiate the latent head. Changed it to two frames so every parameter is checked. |
| 3 | `v4-phase3/lavapipe-cursors` | GPU carry and full interrupted resume on software Vulkan | Both pass; zero validation errors. |
| 3 | `v4-phase3/radv-cursors-border` | 24-frame GPU carry, partial/outside borders, interrupted Adam resume | Both pass; unchanged-weight carry within 1e-5, resumed parameters within 1e-6; zero validation errors. |
| 3 | `v4-phase3/clippy` | Workspace/all-target lint | Two style errors: nested conditional and test-module ordering; corrected. |
| 3 | `v4-phase3/radv-cursors-bindings` | GPU carry and full resume | Resume passes; carry values agree but its border fixture had only partial taps, correctly renormalized as valid. Added explicit wholly-outside and partial-border cases. |
| 3 | `v4-phase3/release-build` | Release trainer | Pass. |
| 3 | `v4-phase3/radv-cursors-space` | GPU cursor carry and full resume | Shader compilation rejected WGSL reserved word `target`; renamed binding. No validation error or training run. |
| 3 | `v4-phase3/initial-unit` | Initial workspace unit tests | 92 pass. |
| 3 | `v4-phase3/cursor-unit` | Crop/gain/prefetch parity, sampler replay, margin exclusion | 94 pass; ten GPU gates run separately. |
| 3 | `v4-phase3/radv-cursors` | First cursor GPU gate attempt | Linking ran out of disk before execution; recorder manifest/log writes also failed. Moved 1.9 GB of regenerable incremental cache to `/tmp`, preserving run/data archives. |
| 3 | `v4-phase3/integrated-check` | Mapped loader, GPU pipeline and true-resume CLI compile | Fixed borrowed checkpoint path and redundant trait import. |
| 3 | `v4-phase3/pipeline-check` | First GPU cursor pipeline compile | Fixed Blade transfer-trait import and fallible fence wait; no GPU run yet. |
| 3 | `v4-phase3/loader-sampler-unit` | First mapped-loader and resumable-cursor compile | Compile rejected conversion of Bytemuck's non-Error cast failure; mapped it to a descriptive error. |
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
