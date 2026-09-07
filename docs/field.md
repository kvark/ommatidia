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
emitters, per-light contribution passes, and calibrated directional environment maps remain extensions; they are not
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

The incident-light experiment below tests additional physical supervision. Shared
or pretrained core weights and material/visibility factorization remain separate
experiments. Keep observed geometry exact in realtime Ommatidia.

Related primary work: [pixelNeRF](https://arxiv.org/abs/2012.02190) conditions a
radiance field on image-aligned features; [NeRFactor](https://arxiv.org/abs/2106.01970)
separates material/visibility/illumination. A common transport prior across these
two observation regimes is the research hypothesis, not an established novelty
claim. OLATverse currently requires registration; it is not a runtime or CI
prerequisite.

## Incident-radiance experiment

`--incident-probes 24 --incident-batches 8` adds training-only angular radiance
labels to static captures (`--field-views` or `--scene-labels`). The scene manifest
becomes version 2; version 1 without these targets still loads. No G-buffer or
light labels are added to `Observations`, `Prepared`, or exported RGB contexts.

Each probe stores the **actual offset world-space origin**, a unit direction
pointing toward the scene, direct and indirect RGB means, and variance of each
mean. Total variance is measured on paired sums, not fabricated by assuming
independent components. Provenance records path depth, four paths per independent
batch, batch count, offsets, ray distance and the canonical radiance ceiling.
Two or more batches are required. Finite sampling variance does not capture
truncation bias or rare paths that were never sampled.

The capture uses a dedicated 1×1 nonjittered Blade canonical renderer: the same
materials, geometry, visibility, environment and path integrator as RGB capture.
It does not alter the image renderers' RNG/history. On a hit, first-hit emission
is direct and the remainder is indirect; on a miss, the environment is direct.
Thus indirect means **at least one scattering event beyond the receiver**. No
receiver albedo, cosine, inverse-square multiplier or `1/pdf` is baked into these
radiance targets. The default 0.01 world-unit offset is recorded, not hidden.
Half the proposals aim at finite emitters when possible, the rest sample a
receiver hemisphere. These are regression strata, not a quadrature rule.

`field --incident-rays 4 --incident-weight 0.1` adds those rays to the existing
volume-rendering graph. It does **not** add an independent incoming-light head:

```
direct   = sum(T * alpha * emission) + T_end * environment
indirect = sum(T * alpha * scattered_radiance)
incident = direct + indirect
```

Image rays and probe rays therefore constrain the same density/visibility,
emission, appearance and shared encoder. `field::graph::build_incident` exposes
all three ray-integrated outputs using ordinary RGB contexts, ray queries and
acquisition bounds. Parameter names/shapes remain compatible with field v1.
The appearance branch still represents captured illumination, not a BRDF or a
light-conditioned relightable transport operator.

The auxiliary loss is the mean of direct/indirect log1p-RGB errors. Bounded
inverse uncertainty weights use the delta-method log variance with a 0.01 floor,
normalize over valid components, and are applied through square-root masks.
Loss weight zero retains the same graph, rays, initialization and random sample
stream; only the masks change. Missing probe metadata is an error when incident
training is requested, not a zero-light label.

Run `bash benchmarks/incident-lavapipe.sh` for the paired weight-zero/0.1 ablation
with two optimization seeds, identical images/probes and fixed update counts.
Two geometries appear under two training lighting seeds; two unseen geometries
use a third lighting seed. Checkpoints are reloaded before scoring held images
and incident rays. This is an unseen-scene test, not a relighting test on the
same scene. Held probe coordinates are evaluator queries, not model inputs
encoding geometry or lighting beyond the requested ray.

The capture test explicitly checks visible emission, unobstructed sky, blocked
emission, reflected light and distance-invariant radiance. Model tests check the
split sum, CPU integration, gradients through visibility and checkpoint reload.
Related derivation: [PBRT surface reflection](https://pbr-book.org/4ed/Radiometry%2C_Spectra%2C_and_Color/Surface_Reflection).
