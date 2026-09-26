# One-week denoising quality goal

## Goal

Deliver one demonstrably useful denoiser at the current **1-spp 128×128 input →
256×256 reconstruction** setting: visibly cleaner surfaces and reflections,
stable in motion, with one architecture and a reproducible checkpoint.

Keep the existing recurrent, lobe-separated residual U-Net as the single model
family. Improve the implementation where diagnosis supports it; do not accumulate
alternative architectures or historical result galleries in the repository.

The latest 4,000-update fine-tune took about 25 minutes, but visible noise remains.
Prioritize diagnosing learning limitations, improving supervision, and testing
over simply extending training.

## Acceptance criteria

These are targets for the week, not claims about current results. Freeze the
benchmark, crops, and metric definitions before selecting a new checkpoint.

- **Visual quality:** target roughly 50% lower reconstruction-error MSE against
  high-sample references on predefined smooth-surface crops, measured in RGB
  after fixed `x/(1+x)` compression and before sRGB conversion or quantization.
  Preserve edges and material texture. Inspect worst frames as well as averages;
  reduced image variance alone is not evidence of successful denoising.
- **Temporal quality:** no increased flicker, ghost trails, or disocclusion
  failures across held-out clips long enough to exercise history accumulation.
  Check both temporal metrics and videos.
- **Numerical quality:** aim for approximately +1 dB on the frozen held-out set,
  measured against the current checkpoint on identical inputs and evaluation
  settings. This is secondary: PSNR improvement alone does not pass the visual bar.
- **Correctness:** pass numerical, gradient, recurrence, and checkpoint-reload
  checks, and resolve the outstanding Vulkan-validation failure. Numerical
  agreement with validation disabled does not establish API conformance.
- **Evidence:** publish one checkpoint, reproducible evaluation, representative
  README comparisons and short videos, and measured GPU time and memory.
  Report remaining failures and correctness limitations explicitly.

## Schedule

Implementation protocol and evidence: [docs/quality-week.md](docs/quality-week.md).
The quality gates below remain open until the final audit and visual review pass.

### Days 1–2: establish what is broken

- [x] Freeze a small benchmark covering diffuse lighting, glossy reflections,
      motion, and disocclusions. Separate development cases from the final audit;
      already-inspected failures are diagnostic cases, not a fresh holdout.
- [x] Test whether the model can fit a tiny, cleanly controlled dataset.
      The corrected model trained on 64-frame sequences is substantially cleaner
      and stable on independent noise for that scene, but overfits the scene.
      This is not a held-out quality pass. See the protocol.
- [x] Isolate diffuse, specular, and temporal-history contributions to the noise.
      Component diagnostics identify diffuse blotches as well as specular noise.
      Resetting every frame loses 4.2–5.0 dB on the tested development cases;
      keep recurrence and improve cold-start supervision. See the protocol.
- [x] Verify capture and supervision correctness, and address the remaining
      GPU-validation failure. Track numerical correctness and API conformance
      separately.
      The pinned compiler correction passes debug GPU checks on RADV and
      LavaPipe, plus capture/train/reload with zero validation errors.

**Day-two checkpoint:** if a simple stationary scene still cannot become clean,
pause scaling training and resolve the bottleneck first.

### Days 3–5: fix and train the existing architecture

- [x] Change the specific input, history, or loss behavior supported by the
      diagnostic tests.
- [x] Add targeted training data where the tests expose a coverage gap.
      Added 20 diverse 64-frame training scenes and 10 disjoint development
      scenes, covering static lighting, camera/object/light motion and assets.
- [ ] Run bounded, hypothesis-driven experiments, selecting checkpoints on the
      development set rather than the final audit.
- [ ] Retain one implementation and record the configuration and provenance of
      the selected checkpoint.

### Days 6–7: evaluate and publish

- [ ] Evaluate the selected checkpoint against the starting checkpoint on the
      frozen audit, including worst cases and temporal behavior.
- [x] Add an established spatial-quality reference such as OIDN, documenting
      input, guide, and resolution differences. OIDN is not temporally stable;
      assess temporal quality separately. See the
      [OIDN documentation](https://www.openimagedenoise.org/documentation.html#rt).
- [ ] Measure GPU time and memory with the evaluation conditions recorded.
- [ ] Update README images, numbers, and short videos with reproducible results.
      State which acceptance criteria passed and which remain unmet.

## Outside this week's commitment

- DLSS parity.
- Broad game generalization.
- Real-time 1080p integration.
- A collection of competing architectures.

Time-box this sprint to seven working days. The quality targets are not a promise
that longer training will reach them. At the end of the week, report the measured
outcome, including failed gates and the diagnosed bottleneck; do not silently
extend the sprint or expand its scope.

Success means an immediately recognizable improvement in both still images and
motion, supported by measurements. If the quality gates are not met, do not
promote an inadequate checkpoint or claim the quality goal is complete.
