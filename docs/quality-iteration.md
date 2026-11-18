# F1 quality iteration

## First question

Does showing the same surface under a second independent 1-spp realization
help F1 suppress noise on **unseen scenes**? Each scene in the 200-scene corpus
has one recorded noise trajectory. A network can learn the errors in that
trajectory. Changing only training noise lets us test that explanation.

## Experiment fixed before capture, 2026-11-18

1. Use the current Meganeura/Blade pins and the single 16-channel, four-level
   F1. Capture procedural scenes on RADV with its existing Mesa 26.2.3 setup.
   Quality runs can share the RX 7900 XT. Leave other jobs running.
2. Capture eight training and eight development scenes. Each split has four
   static scenes and four with camera, object and light motion. Base seeds in
   that order are 18011001/18012001 for training and 18021001/18022001 for
   development. Use 24 jittered frames per scene, 64² 1-spp inputs, 128²
   output guides, 1,024-spp references, eight bounces, canopy, gloss, texture
   and eight ground patches. Moving cases use 0.03 camera and object motion
   and 0.05 light motion. Keep every frame.
3. For each training case render a second stream with LR path-frame offset
   512. Check high-resolution targets/guides and the observed input geometry
   for equality, the LR radiance for differences, and actual seeds for split
   overlap. The control lists the first two captures twice. The challenger
   substitutes the second noise stream in the last two slots. This preserves
   scene ordering, sampling probabilities, window count and target means.
4. Train both from scratch with seed 1, 2,000 updates, two 32² crops per
   update, four-frame unroll and the same default objective, augmentation and
   learning-rate schedule. Score only the final checkpoints. Compare paired
   development sequences under causal and reset-every-16 evaluation. Report
   PSNR, official FLIP, temporal error, cold/early/warm behavior, energy and
   each moving sequence's final frame. Use 1,000 paired whole-sequence
   bootstrap draws (seed 31). Show static sequence 0 and moving sequence 4,
   frames 0 and 23, at the same display transform.
   On a second causal development noise stream (offset 512, identical HR
   targets), estimate noise sensitivity as half the mean squared difference
   between the model's two predictions in compressed RGB. Subtract this from
   the mean of the two reference errors to estimate the repeatable component.
   Report both so that merely blurring the image cannot count as denoising it.
5. Advance a data change if causal warm PSNR improves (95% lower bound above
   zero), mean causal FLIP falls, and cold PSNR loses at most 0.1 dB in both
   protocols. Repeat the pair with seed 2 if the warm interval crosses zero.
   If it still overlaps, report uncertainty and keep the current recipe.
   Inspect the lobe and temporal errors for the next hypothesis. Do not select
   a checkpoint or promote the gallery on this small panel.

The next milestone is the 60k F1 evaluation on the original hashed 200-scene
corpus and ten development sequences against the learned v3 control. Check
recovered captures, checkpoint and control against their recorded identities.
If they cannot be recovered, qualify regenerated inputs by hash. Any newly
captured benchmark requires a matched control before making a quality claim.
The frozen [Decision A criteria](../PLAN.md#9-phase-5-training-ladder-and-decision-point-a-3-days)
apply there. Confirmation stays reserved for the selected final model.

## Fixed follow-up on training duration

The first pair improved warm PSNR but missed the mean FLIP gate. Keep the
single-stream input. Before running the follow-up, fix a single 12,000-update
fit from scratch, with the same seed, sampler, data and objective as the
2,000-update control. The planned cosine schedule stretches with the budget.
Score only the last update against the original control, with the same held-out
noise pair, image selections, metrics, bootstrap and advancement thresholds.
Record whether the reference-relative gradient error and the repeatable RGB
error improve as well. This tests the benefit of more optimization on eight
training scenes. The official corpus and the v3 comparison remain separate.
