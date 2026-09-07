# Native transport: paired loss-normalization ablation

**Spatial improvement over relative weighting, not a quality promotion.**
Fixed exposure remains opt-in. Temporal error rises versus the learned control;
both learned arms lose energy and broad lighting fidelity to the fixed mixture.

[Completed LavaPipe run 34148548088](https://github.com/kvark/ommatidia/actions/runs/34148548088),
source `0151ae9097415e803bb4eced4c452a77625689f1`, artifact
`normalization-lavapipe-quality`. Reproduce:

```
bash benchmarks/normalization-lavapipe.sh
```

## Protocol

Eight fitting scenes × eight frames, capture seed 7; four fitting-disjoint
validation scenes × sixteen frames, fresh seed 50000. The latter was chosen
before running the pair, after inspecting the earlier seed10000 study. No
validation score selects an update or changes preset hyperparameters. These
are procedural seeds, not four unrelated real-mesh families.

Both arms use the same data, graph, initialization recipe, two-frame unroll,
eight channels, 512 updates, optimization seed 7 and initial rate 0.001 with
the same cosine schedule. Inputs are independent 1-spp 32×32 paths with 64×64
surfaces/output, matching eight-bounce transport, camera/object/light motion,
jitter and split radiance. References: 256 spp fitting, 512 spp validation.
The regenerated fitting OMD matches the first native study byte-for-byte.
Saved weights are reloaded before native full-sequence evaluation.

The only explicit training change is loss-scale input `1/(0.1+target_lobe)`
versus fixed configured exposure (1), affecting physical, low-frequency and
temporal errors. Primary compressed error, confidence loss and global loss
coefficients remain unchanged. Learned recurrent states naturally diverge
between optimized models. This is not a fully unbiased loss.

## All 64 validation frames

The deterministic reports match exactly between arms: 64 outputs, 60 temporal
transitions and four resets. Higher PSNR/SSIM is better; ratios target 1.

| Metric | Fixed candidates | Relative loss | Fixed exposure |
|---|---:|---:|---:|
| PSNR | 22.3100 dB | 21.6528 dB | 22.2735 dB |
| SSIM | 0.66016 | 0.67746 | 0.68651 |
| Low-frequency PSNR | 29.0738 dB | 25.1368 dB | 26.5783 dB |
| Linear energy ratio | 0.99786 | 0.93598 | 0.95318 |
| Detail ratio | 1.02604 | 0.93760 | 0.94259 |
| Relative MSE | 0.58932 | 0.41393 | 0.42052 |
| Temporal MSE | 0.00351871 | 0.00223232 | 0.00253867 |
| Reset PSNR | 19.0998 dB | 19.4515 dB | 19.4547 dB |

PSNR/SSIM use `x/(1+x)`. Low-frequency error uses 8×8 blocks of compressed
RGB, not linear-energy pooling. Energy is scene-linear. Frame means have equal
weight; frames within a sequence are correlated.

| Validation geometry | Fixed PSNR | Relative PSNR | Exposure PSNR | Exposure minus relative |
|---|---:|---:|---:|---:|
| 50000 | 22.9841 | 22.1760 | 23.0300 | +0.8540 dB |
| 2654452457 | 22.5564 | 21.4761 | 22.2833 | +0.8072 dB |
| 5308821538 | 21.0570 | 20.3180 | 20.8059 | +0.4879 dB |
| 7963324027 | 22.6425 | 22.6411 | 22.9747 | +0.3336 dB |

Fixed exposure gains 0.6207 dB spatial and 1.4415 dB low-frequency PSNR over
relative weighting, improving spatial quality in all four sequences. Energy
moves toward 1 but remains 0.9532. Temporal error is 13.72% worse than the
relative arm, although 27.85% lower than the deterministic mixture. There is
no all-metric win. Inspection of every mature frame confirms remaining mottling
and detail loss; the output is not a demonstrated replacement for SVGF.

Independent reference ranges on the first sequence give 30.8372 dB pairwise,
an estimated 33.8475 dB one-reference noise floor, and energy B/A 0.999932.
Independent NumPy replay gives 49.3650 dB agreement at matching 8×8 blocks;
the reference-noise tool's 16×16 value is 55.2935 dB. Noise-floor inference
assumes independent, equal-variance samples; it does not test shared bias.

## Decision

Retain the opt-in flag and reproducible pair; promote neither asset. Target-based
normalization plausibly contributes to darkening, and this intervention improves
the spatial tradeoff, but does not isolate every source of estimator/loss bias.
Next isolate the remaining compressed objective and confidence/temporal terms
on construction controls, then audit the selected recipe on fresh scenes and
real-mesh families. Keep energy/detail gates alongside temporal smoothness.

[First native study](transport-lavapipe-2026-09-07.md) · [Roadmap](../quality-roadmap.md).
