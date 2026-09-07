# Experimental radiance-field variant

Posed linear RGB and acquisition bounds in; a queryable density/radiance field
out. No G-buffer, velocity, true light parameters, target depth, or scene-index
embedding enters inference. This is an image-conditioned model trained across
captures, not a separate free parameter table fitted to each scene.

## Architecture

```
RGB + camera rays -> RGB adapter -> shared local image pyramid
    -> project each 3D query into source views
    -> masked feature mean, variance, coverage + position encoding
    -> geometry latent -> density, source emission
                      + viewing direction -> scattered radiance
pooled source features -> environment
```

`neural.rs` is the actual common core used by `transport::graph` and
`field::graph`. Different adapters/heads have distinct parameter names. Equal
widths have compatible core shapes; the default field is wider (64 channels,
128-wide query MLP) than the 8-channel realtime model. Sharing implementation is
not evidence that separately trained weights transfer, or that a shared latent
has learned physical transport. Joint training/distillation is a later test.

`build_points` evaluates density, total radiance, emitted radiance, and a constant
RGB environment. Density and emission never receive query viewing direction.
`build_render` adds differentiable front-to-back volume rendering, including
residual background transmittance. Density is per normalized scene unit; ray
steps are scaled accordingly. Frustum membership is **not occlusion visibility**.
Mean/variance pooling is the initial baseline, not a learned visibility solver.

Radiance is emission plus nonnegative, view-dependent appearance. The appearance
head is not yet a BRDF or a relightable transport operator. Source supervision
helps identify emitters; it does not make albedo, visibility and lighting uniquely
recoverable. This version reconstructs the *captured illumination*. It cannot
render an arbitrary new illumination from a supplied light list.

## Capture

```
cargo run -p ommatidia-data -- --out data/field.omd --samples 8 \
  --field-views 8 --lr 32x32 --scale 2 --canonical-frames 256 --seed 7
```

`--field-views` captures static, centered camera orbits and writes RGB-only OMD
records plus `.scene.json`. Labels refer to HR RGB and its exact camera, not
jittered LR observations. `--scene-labels` annotates ordinary static procedural
captures. Unsupported catalog lights, copied references, object/light motion,
and jitter are rejected rather than mislabeled. Inference needs camera poses;
recovering unposed cameras is outside this experiment.

Each scene label contains:

| Quantity | Representation |
|---|---|
| Local sources | Exact emitter mesh vertices/triangles and linear emitted RGB radiance |
| Environment | Constant RGB radiance, matching the renderer's unit-white fallback |
| Light supervision | Surface emission probes on both emitters and non-emissive surfaces |
| Provenance | Geometry seed, independent lighting seed, sample-to-camera mapping |
| Acquisition coordinates | Declared center/radius, supplied at inference too |

Radiance is in **renderer RGB units**, not watts/lumens or calibrated spectra.
Emitter radiance is not total power, and the finite source mesh must not be
collapsed into a distant direction. These probes supervise material emission on
surfaces, not density or emitted energy per unit volume.

`--lighting-seed N` changes only source RGB/intensity. Repeat a capture with the
same geometry seed and a different lighting seed to obtain controlled relighting
pairs without changing camera or geometry. Environment variation, textured
emitters, per-light contribution passes, direct/indirect incident-radiance probes,
and calibrated directional environment maps remain extensions; they are not
fabricated from final RGB. OLAT mixtures are valid only for the same geometry,
camera, linear exposure and background accounting. Existing `transport::olat`
can form finite-light Monte Carlo pairs, not arbitrary path-tracing noise.

## Training and evaluation

```
cargo run -p ommatidia-train --bin field -- --data data/field.omd \
  --image 64 --views 3 --out target/field --steps 256
```

Repeated `--data` adds scenes/lighting conditions. The deterministic split uses
context cameras, separate fitting cameras, and the final held camera. All query
ray samples come from the acquisition cube, never target depth. Auxiliary
emission probes are balanced between sources and known non-emitters. Labels are
fed through `Targets`, separately from `Observations`/`Prepared`.

The loss combines log-radiance image error with masked emitted-radiance and
constant-environment supervision (0.05 each). All image-encoder, query-decoder
and volume-rendering operations participate in autograd. Probe masks normalize
by valid supervision count, not the total ray-sample count.

Fixed-budget checkpoints are saved and reloaded **before** held-camera scoring;
held quality never chooses an update. `--eval-data` additionally requires unseen
geometry seeds, even when illumination differs. Reports distinguish held cameras
of fitting scenes from new scenes. They include black, context-mean and untrained
controls, linear/log error, named compressed PSNR, and images. No LavaPipe speed
numbers are product claims.

Saved weights/config plus `*-context.json` form a reusable image-conditioned
field. Context exports contain only RGB, cameras and acquisition bounds, not
lighting labels. `build_points` + `Prepared` query arbitrary positions/directions.
There is deliberately no blade-volume integration.

## Gates

First pass camera roundtrips, label isolation, view permutation invariance,
view-independent density, CPU/GPU volume integration and backward/reload tests.
Then run `bash benchmarks/field-lavapipe.sh`. Its tiny budget checks complete
capture/training/serialization/held-view evaluation; it cannot establish robust
geometry, high-quality relighting, or beneficial cross-task transfer.

The next scientific comparison is identical geometry under independently varied
light, with a field-only versus shared/pretrained-core control and independent
scene/camera/light holdouts. Add supervised incident-radiance probes and a
material/visibility factorization only after the basic field beats its image
baselines. Keep observed geometry exact in realtime Ommatidia.

Related primary work: [pixelNeRF](https://arxiv.org/abs/2012.02190) conditions a
radiance field on image-aligned features; [NeRFactor](https://arxiv.org/abs/2106.01970)
separates material/visibility/illumination. A common transport prior across these
two observation regimes is the research hypothesis, not an established novelty
claim. OLATverse currently requires registration; it is not a runtime or CI
prerequisite.
