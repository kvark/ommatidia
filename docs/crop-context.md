# F1 training crop context

## Matched experiment

Use the two existing single-stream, 12,000-update F1 controls from the
[quality iteration](quality-iteration.md). Train two candidates from scratch
with seeds 1 and 2 on the same eight scenes. The sampler still draws two
32×32 LR target windows per update and four frames per window. Expose 48×48
LR pixels to the candidates. Shift the extra context toward the interior
near physical edges, in multiples of eight LR pixels to preserve both the
encoder stride phase and the 4×4 low-frequency loss grid. Keep the original
four-output-pixel margin on artificial target edges and supervise real image
edges. Check target values, masks, normalization, cursor choices and the
per-update count of supervised pixels against the controls.

Evaluate the final checkpoint on the ten restored development sequences
under continuous history and resets every 16 frames. Pair by scene and
reference hash, using the same 1,000 whole-sequence bootstrap draws (seed 31).
Measure FLIP, PSNR by reset age, the 40 fixed crop/frame selections, temporal
MSE, energy and moving-light final frames. Use the separate eight-scene
paired-noise panel for noise sensitivity, the original cold-frame center
crop probe for context sensitivity, and the same four preselected pictures.

Advance to the full corpus if both seeds lower causal FLIP, improve warm PSNR
with a positive 95% lower bound and lose at most 0.1 dB of cold PSNR under
either protocol. Check cold and warm smooth-crop MSE, temporal MSE and
energy before accepting any tradeoff. On a miss, test distinct scene count
at the same 12k budget with the original 32×32 context and seed pair: replace
the duplicate input captures with four new static and four new moving scenes
(base seeds 18031001 and 18032001) at the original capture settings.

To reproduce the training change, add `--context 8` to the control's
`transport --crop 32` command. The radius is in LR pixels per side.
It must be a multiple of the encoder and pooling stride and leave room for
every sampled stride phase. The default radius is zero. A training resume
uses the original radius and restores the full input window's recurrent
state. Inference takes the same F1 checkpoint and graphics inputs as before.
