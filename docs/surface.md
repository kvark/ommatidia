# Surface supervision and fitting diagnostics

An opt-in field experiment, not a change to realtime inputs or published weights.

## Contract

`ommatidia-data --field-views 8 --surface-labels` writes manifest v3 with a v1
`surface` record for every camera. It contains f32 Euclidean first-hit distance
along the unit camera ray (`null` for a certified miss), first-hit emitted RGB,
the maximum certified distance, and `pixel_filter: "center"`. Older manifests
without surface targets remain valid. Existing datasets are not relabeled.

RGB and geometry must describe the **same ray**. Enabling these labels captures
nonjittered, pixel-centred RGB as well as geometry. The old antialiased RGB can
mix foreground/background while centre-ray depth hits only one; it is not a
valid target for this loss at silhouettes. RGB-only OMD records remain RGB-only:
the extra GPU readbacks are stripped after constructing the target manifest.
These maps come from Blade's actual first-hit query and primary emission, not
an approximate CPU mesh tracer. Unsupported dynamic/catalog/copy-reference
capture combinations retain the existing rejection rules.

## Loss

The existing field produces density. Along a ray its termination probability is

```
P(hit interval k) = exp(-sum(j<k, sigma[j]*delta[j])) * (1-exp(-sigma[k]*delta[k]))
P(escape)         = exp(-sum(j, sigma[j]*delta[j]))
```

`--surface-weight W` adds `-W * log(P(label)+1e-8)`, averaged over valid camera
rays. A hit labels its containing sample interval; a miss labels escape. This
constrains empty space **before** a first hit, without declaring occluded space
behind it empty. A hit outside acquisition bounds, or a miss not certified as
far as the ray/cube exit, is masked. Incident rays have no implicit surface label.
No target-derived depth is used to place samples. Position, direction, bounds
and the same fixed midpoint schedule produce identical queries in both arms.

The interval is a finite-resolution surface constraint, not exact reconstructed
geometry. Narrow objects may still need denser or predicted hierarchical
sampling. The first version uses hard bins rather than a noise-derived depth
uncertainty distribution. Missing labels are errors when explicitly requested.
Zero weight retains the identical graph, initialization and random sample stream.
Inference graphs have no `target.*` inputs, and parameter names/shapes are unchanged.

`--emitter-fraction 0.25` reserves a quarter of fitting pixels for visible
emitters where available. Their first-hit support receives the same termination
loss. Other pixels remain uniformly sampled. This deliberately reweights the
training objective; it is **not** an unbiased estimate of uniform image error.
The paired control uses the same stratum. Hidden emitters are not automatically
assigned density by an unobserved surface target.

## Diagnose before scaling

```
bash benchmarks/surface-lavapipe.sh
SURFACE_RESULTS=target/surface-long FIELD_UPDATES=2048 bash benchmarks/surface-lavapipe.sh
```

`FIELD_IMAGE` independently selects the image extent. Use separate result
directories when comparing settings. The recipe has one construction scene,
eight cameras, three source views, four fitting views and a separate validation
camera. It is **not** an unseen-scene benchmark. Compare fixed update counts;
no held score picks a checkpoint.

`field --diagnostics` saves fitting/held RGB, conditional expected-depth maps,
reference depths, opacity, and controls that zero RGB or reassign RGB images to
unchanged camera poses. It records termination likelihood, early termination,
hit/miss accuracy, world-space depth MAE and probability at visible emitter
surfaces. Out-of-bounds/uncertified rays are excluded and counted. Depth MAE is
conditional on a hit prediction; inspect opacity and termination likelihood too.
Depth previews use depth/acquisition-radius and the same display transform for
prediction/reference. A black reference pixel denotes a miss. These are
illustrations, not calibrated grayscale depth measurements.

RGB controls leave cameras/bounds fixed. They diagnose dependence on observations,
not correct geometry: zeroing is out of distribution, and fitting a single scene
can memorize position. Separate scenes and camera configurations are needed to
establish a reusable reconstruction prior. Training targets never enter exported
`Observations` or RGB contexts.

The 512-update default processes 16,384 camera rays over 16,384 fitting pixels,
with repeated and emitter-biased sampling. It is a bounded construction diagnostic,
not a claim of convergence. Scale exposure and capacity independently before
concluding a representation has saturated.

Related prior: [DS-NeRF](https://www.cs.cmu.edu/~dsnerf/) supervises ray-termination
distributions using depth. This implementation uses exact synthetic centre-ray
hits rather than sparse noisy SfM points. Incident supervision remains a separate
opt-in loss; shared weights, arbitrary relighting and blade-volume integration
are outside this change.
