# Native multiscale transport: first expanded LavaPipe run

**Not promoted:** temporal stability improves, but brightness and broad lighting
fidelity regress. These are new native-path measurements, not historical weights.

[Completed run 34146870562](https://github.com/kvark/ommatidia/actions/runs/34146870562),
source `8767d8c`, artifact `transport-lavapipe-quality`. Reproduce with:

```
TRANSPORT_UPDATES=512 TRANSPORT_TRAIN_SCENES=8 TRANSPORT_HOLD_SCENES=4 \
  bash benchmarks/transport-lavapipe.sh
```

Eight fitting scenes x eight frames (seed 7), four fitting-disjoint scenes x
16 frames (seed 10000). Procedural canopy, textures, gloss, camera/object/light
motion, projection jitter, separate radiance lobes and full-resolution surfaces.
Independent 1-spp input, 32x32 -> 64x64 output, matched transport, 256-spp fitting
and 512-spp evaluation references. Core8, two-frame BPTT, 512 optimizer updates,
optimization seed7. Fixed budget, checkpoint reload before full-sequence scoring.
No real-mesh generalization, SVGF comparison, or production-GPU timing is claimed.

## Native quality

Equal frame averages over 64 outputs, 60 temporal transitions and four resets.
The baseline is the native deterministic candidate mixture, not SVGF or a
separately traced input. Both paths receive the same captured observations.

| Metric | Deterministic | Learned |
|---|---:|---:|
| PSNR | **27.2067 dB** | 27.0332 dB |
| SSIM | 0.79842 | **0.80967** |
| Low-frequency PSNR | **34.8553 dB** | 32.4520 dB |
| Linear energy ratio | **1.00674** | 0.96925 |
| Detail ratio | **0.91664** | 0.86136 |
| Relative MSE | 0.29934 | **0.22638** |
| Temporal MSE | 0.00213658 | **0.00134798** |
| Reset-frame PSNR | 24.5883 dB | **25.0444 dB** |

PSNR/SSIM use x/(1+x); energy is scene-linear. Low-frequency error averages
8x8 blocks of compressed RGB; it is not a direct linear-energy metric.
Per-sequence PSNR changes are -0.2135, -0.0826, +0.1868 and -0.5844 dB.
All frames are retained in the artifact. Visual inspection of all four mature
frames shows smoothing and energy loss; one view is mostly occluded, so four
sequence seeds do not constitute broad content coverage.

An independent reference range on the first sequence gives 31.61 dB pairwise,
an estimated 34.62 dB one-reference noise floor, and energy B/A 0.999775.
At the matching 8x8 block extent, reference agreement is 49.47 dB; the existing
reference-noise tool's coarser 16x16 block report is 54.39 dB. The 8x8 value was
computed from the retained OMD references with an independent NumPy replay.
The floor estimate assumes independent equal-variance samples; it cannot diagnose
shared estimator bias. Broad lighting disagreement is much larger than measured
broad reference noise at either block extent.

## Next isolated test

The supposedly physical loss normalizes each pixel by **1/(0.1 + target)**;
the low-frequency term pools after that normalization. It therefore does not
measure unweighted block energy. In the simple equally probable 0/4 target
example, squared relative error prefers 4/1682 rather than the linear mean 2.
This is a possible mechanism, not proof of the whole observed regression.

`transport --fixed-exposure-loss` substitutes the fixed configured exposure for
those physical/low-frequency/temporal loss scales. Inference, parameter shapes,
primary compressed loss, confidence term, graph and random sample stream stay
unchanged. It is an opt-in loss-normalization ablation, not a full removal of
all compressed-loss bias. The paired recipe uses a fresh evaluation seed 50000;
the inspected seed10000 set is now development data.

[Paired recipe](../../benchmarks/normalization-lavapipe.sh) · [Roadmap](../quality-roadmap.md)
