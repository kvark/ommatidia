# Ommatidia architectures

Two experimental heads share the local image pyramid in `neural.rs`.

| Path | Observations | Output | Training |
|---|---|---|---|
| `transport` | Sparse radiance, surface data, motion, recurrent state | Diffuse illumination + specular radiance, remodulated with exact albedo/emission | Full causal sequences, multiscale candidates, confidence labels, short radiance BPTT |
| `field` | Posed linear RGB, acquisition bounds | Queryable density/radiance/emission field and environment | Multiview volume-rendered RGB; target-only synthetic light supervision |

The realtime path does not infer geometry it already knows. The field path
receives no G-buffer or velocity and is allowed a wider/slower network. Shared
code does not yet imply shared trained weights or proven transfer. Neither path
is a demonstrated DLSS replacement. See [field.md](field.md) for the offline
experiment and [quality-roadmap.md](quality-roadmap.md) for the research question.

`transport::Frame` and `field::Observations` contain observations only. Ground
truth uses separate `Target`/`Targets` types. State and rendered values remain
linear; bounded feature encodings are not averaging spaces.

The published v0.3.1 checkpoint remains supported by `ModelConfig`, `Upscaler`
and the C ABI. Their serialized legacy interpretation is unchanged. Do not edit
sidecars to convert old weights into either new architecture. Historical designs
and commands describe their recorded revisions, not the active training tools.

Validation is in `tests/transport.rs`, `tests/field.rs`, the legacy GPU runtime
suite, and the two LavaPipe benchmark scripts. Quality promotion requires
independent held scenes, cameras and illumination, not just a falling fit loss.
