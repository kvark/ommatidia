# Temporary compiler correction

`naga-workgroup-layout.patch` applies to Naga revision
`323acfb729a00c3030e362faa27252afe6368792`. Run
`python3 scripts/prepare-naga.py` before invoking Cargo. It checks out that
revision under ignored `target/patched-naga`, applies the patch, and rejects
unrelated changes. Cargo's cache and the user's sibling checkouts are unchanged.

The unpatched compiler emits `ArrayStride` and member layout decorations for
workgroup memory, triggering `VUID-StandaloneSpirv-None-10684`. Vulkan permits
those decorations on host buffer layouts, but not ordinary workgroup storage.
See [Shader Interfaces](https://docs.vulkan.org/spec/latest/chapters/interfaces.html).
This patch creates undecorated workgroup composite types, preserves buffer
layouts, converts whole-value loads/stores and handles zero initialization.
It does not disable validation or change the denoiser architecture.

The targeted compiler regression covers nested arrays/structures/matrices used
in both workgroup and storage/uniform buffers, dynamic checked accesses,
whole-composite copies, atomics, workgroup-uniform loads, writer reuse, SPIR-V
1.3/1.4 and all three initialization modes. Run with `spirv-val` installed:

```sh
cargo +stable test --manifest-path target/patched-naga/Cargo.toml -p naga \
  --features wgsl-in,spv-out --test naga spirv_workgroup_layout \
  --locked -- --include-ignored
```

Keep this patch isolated and remove the override when a pinned upstream revision
provides the fix. It is not an upstream-accepted change or a claim that the full
wgpu test/CTS matrix passes. Project GPU conformance evidence is recorded
separately in the quality-sprint protocol.
