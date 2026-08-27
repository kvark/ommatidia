# Independent canonical reference noise — 2026-08-27

## Question

Is the remaining model error partly a finite-sample artifact in Blade's
canonical target?

## Protocol

Two files contain the same two seed-72000 scenes, 256×256 targets, eight-bounce
paths, and 4,096 accumulated frames at four paths per pixel. The second run
discarded 4,096 frames before accumulation, so the path sets do not overlap.
Low-resolution records match byte for byte.

```sh
target/release/ommatidia-data \
  --out data/reference-noise-a.omd --samples 2 --lr 128x128 --scale 2 \
  --canonical-frames 4096 --canonical-bounces 8 --input-frames 1 \
  --canopy --ground-patches 8 --textures --gloss --split-radiance \
  --seed 72000 --hr-gbuffer

target/release/ommatidia-data \
  --out data/reference-noise-b.omd --samples 2 --lr 128x128 --scale 2 \
  --canonical-frames 4096 --canonical-bounces 8 \
  --reference-sample-offset 4096 --input-frames 1 \
  --canopy --ground-patches 8 --textures --gloss --split-radiance \
  --seed 72000 --hr-gbuffer

target/release/reference-noise \
  --a data/reference-noise-a.omd --b data/reference-noise-b.omd
```

## Result

| measure | independent reference pair |
|---|---:|
| pairwise PSNR | 48.66 dB |
| estimated one-reference noise floor | 51.67 dB |
| SSIM | 0.992037 |
| relative MSE | 0.00054802 |
| 16×16-block low-frequency PSNR | 73.14 dB |
| mean energy B/A | 1.000024 (+0.002%) |
| detail B/A | 0.999700 |

Two equal-variance independent estimates differ with variance `2 sigma²`, so
the reference contribution to model-versus-reference MSE is half the measured
pairwise value, or 3.01 dB above pairwise PSNR.

## Decision

The canonical reference is not limiting current 30–33 dB reconstructions, and
its low-frequency noise is especially negligible. Continue treating the
visible broad variation and structural deficit as model/input failures. The
0.992 pairwise SSIM is useful context near convergence, but it cannot explain
the present 0.89–0.95 model range.
