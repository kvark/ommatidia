# Catalog B8 residual over split-lobe — 2026-08-26

## Question

Does adding held-out ABO objects to training reverse the SSIM loss of a B8
residual over the fixed split-lobe estimator? The 24-scene direct-lobe probe
improved PSNR and lost SSIM; the roadmap asked to grow the corpus before
restoring that head or widening the network.

## Protocol

- Train: `data/catalog-abo-train.omd`, 24 sequences × 16 1-spp jittered frames,
  seed 80000. Each scene places two catalog glTFs in the procedural room.
- Hold-out: `data/catalog-abo-holdout.omd`, 8 sequences, seed 81000. Sidecar
  ids do not overlap the train set.
- Targets: 16,384-spp eight-bounce references copied with `--reference-from`.
- Model: direct B8, 3 levels, 1 block, 78,920 parameters, 18.2 GFLOP/1080p.
  `--prediction subpixel --reconstruction-base split-guided --history-frames 16
  --temporal-features phase-lobes`. Cosine 3e-4 → 1e-5, 4000 steps, seed 0.
- Score: 480 non-reset 64×64 crops on the hold-out (15 ages × 8 scenes × 4
  crops). Linear PSNR/relMSE; display SSIM and detail; 16×16-block
  low-frequency PSNR.

HSSD stage interiors were captured but not trained: every split peaked at
radiance 1.00 (furnace-lit empty shells).

## Hold-out versus the frozen estimator

| step | network PSNR | vs split guide | SSIM | split SSIM | LF PSNR | relMSE | worst crop |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1000 | **33.12 dB** | **+0.20 dB** | 0.9156 | **0.9507** | **37.30** | **0.02405** | 0.25 |
| 2000 | 33.01 dB | +0.10 dB | 0.9002 | 0.9507 | 37.15 | 0.02296 | 0.15 |
| 3000 | 32.93 dB | +0.02 dB | 0.8923 | 0.9507 | 36.97 | 0.02348 | 0.16 |
| 4000 | 32.90 dB | −0.01 dB | 0.8891 | 0.9507 | 36.93 | 0.05416 | **4.65** |

Split guide itself is 32.91 dB / 0.9507 SSIM / 79% detail / 37.08 dB
low-frequency on this set. The residual's best PSNR is the same order as the
old 24-scene compact residual (+0.14 dB). SSIM never catches the estimator.
Detail rises (79% → 92–96%) while structure falls. After 1000 steps the fit
keeps sharpening and eventually invents dark-crop error (worst relMSE 4.65).

Temporal deltas at 1000 steps are small and real: +0.10 dB reprojected, +0.29
dB low-frequency temporal. They survive to 4000 steps even as radiometry
degrades, so stability is not the thing the extra objects failed to teach.

The written checkpoint is the 4000-step stem
`runs/catalog-b8-split.{safetensors,ron}`. Intermediate evals overwrote it;
step 1000 was the only one that beat the estimator on PSNR.

## Decision

Do not promote this residual and do not restore a direct-lobe head from it.
ABO object identity was held out correctly; more of the same residual on
richer meshes still trades SSIM for a few tenths of a decibel, then overfits.
The fixed split-lobe estimator remains the quality floor on this corpus.

The next data change that could matter is furnishing HSSD (compose
`scene_instance.json` plus lights) or adding a third *procedural* seed family
with independent 16,384-spp targets — not B16 on these weights.
