# One-week denoising quality goal

## Goal

Deliver one demonstrably useful denoiser at the current **1-spp input, 2×
reconstruction** setting: visibly cleaner surfaces and reflections, stable in
motion, with one architecture and a reproducible checkpoint.

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
  high-sample references on predefined smooth-surface crops, without smearing
  edges or material texture. Inspect worst frames as well as averages; reduced
  image variance alone is not evidence of successful denoising.
- **Temporal quality:** no increased flicker, ghost trails, or disocclusion
  failures across held-out clips long enough to exercise history accumulation.
  Check both temporal metrics and videos.
- **Numerical quality:** aim for approximately +1 dB on the frozen held-out set,
  measured against the current checkpoint on identical inputs and evaluation
  settings. This is secondary: PSNR improvement alone does not pass the visual bar.
- **Evidence:** publish one checkpoint, reproducible evaluation, representative
  README comparisons and short videos, and measured GPU time and memory.
  Report remaining failures and correctness limitations explicitly.

## Schedule

Implementation protocol and evidence: [docs/quality-week.md](docs/quality-week.md).
The quality gates below remain open until the final audit and visual review pass.

### Days 1–2: establish what is broken

- [ ] Freeze a small benchmark covering diffuse lighting, glossy reflections,
      motion, and disocclusions. Separate development cases from the final audit;
      already-inspected failures are diagnostic cases, not a fresh holdout.
- [x] Test whether the model can fit a tiny, cleanly controlled dataset.
      Initial fit improves RGB but remains visibly noisy and worsens specular
      decomposition; the diagnostic gate has not passed. See the protocol.
- [ ] Isolate diffuse, specular, and temporal-history contributions to the noise.
- [ ] Verify capture and supervision correctness, and address the remaining
      GPU-validation failure. Track numerical correctness and API conformance
      separately.

**Day-two checkpoint:** if a simple stationary scene still cannot become clean,
pause scaling training and resolve the bottleneck first.

### Days 3–5: fix and train the existing architecture

- [ ] Change the specific input, history, or loss behavior supported by the
      diagnostic tests.
- [ ] Add targeted training data where the tests expose a coverage gap.
- [ ] Run bounded, hypothesis-driven experiments, selecting checkpoints on the
      development set rather than the final audit.
- [ ] Retain one implementation and record the configuration and provenance of
      the selected checkpoint.

### Days 6–7: evaluate and publish

- [ ] Evaluate the selected checkpoint against the starting checkpoint on the
      frozen audit, including worst cases and temporal behavior.
- [ ] Add an established spatial-quality reference such as OIDN, documenting
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

The final result should be an immediately recognizable improvement in both still
images and motion, supported by measurements. If the quality gates are not met,
report that outcome and the diagnosed bottleneck without claiming completion.
