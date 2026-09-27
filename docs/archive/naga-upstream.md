# Phase 0: upstream compiler port for owner review

Prepared against `kvark/wgpu` branch `claude/fix-spirv-layout-decoration-Bn4Oc`,
commit `731fd87244c891f19c53b77826731fe809a32f9b`, the current head of
[wgpu #9295](https://github.com/gfx-rs/wgpu/pull/9295).
Local checkout: `target/naga-pr9295`. No upstream commit/push has been made.

Reviewable patch: `runs/v4-phase0/naga-pr9295-pre14.patch`, SHA-256
`585edd49b484befe96c1e3fa99541fd784a0a23d06809f474ac75deee3198e02`.
It applies cleanly to the PR head and contains the code, regression test,
changelog and three affected SPIR-V snapshots with their GLSL sidecars.

The port keeps undecorated workgroup types on every SPIR-V version and converts
whole composite values using recursive extract/construct before 1.4. Version
1.4+ keeps `OpCopyLogical`; the [SPIR-V specification](https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html#OpCopyLogical)
does not provide that instruction on earlier targets. This addresses the
reviewer's question without accepting invalid layout decorations on older targets.

Checks (full commands and tool/source snapshots in `runs/v4-phase0/`):

- Formatting and all-target/all-feature Naga Clippy, warnings denied: pass.
- Six generated modules (1.3/1.4 × three initialization modes), nested shared
  buffer/workgroup types, whole copies, checked dynamic access, matrices,
  atomics, uniform loads and writer reuse: both targeted tests pass, including
  `spirv-val`. `OpCopyLogical` is absent below 1.4.
- Full Naga suite: 139 unit, 234 integration and six doctests pass. It initially
  failed because `spirv-cross` was missing; an isolated Ubuntu package fixed
  that. The unmodified PR head was tested with the same tool to identify and
  exclude five tool-version-only sidecar changes from the patch.
- All three changed shader snapshots assemble and pass Vulkan SPIR-V validation.
- Full `cargo xtask test` initially could not run without `cargo-nextest`;
  with an isolated SHA-verified binary, 1,915 tests pass, nine fail and 19 are
  skipped. Eight failures require the absent `dxc`; one OpenGL expected-failure
  test unexpectedly succeeds. A matched rerun of those families on the
  unmodified PR head reproduces all nine failures (27 pass). The port does not
  introduce those failures, but the full suite is **not green**. No CTS pass or
  upstream CI pass is claimed.

The upstream [contribution policy](https://github.com/gfx-rs/wgpu/blob/731fd87244c891f19c53b77826731fe809a32f9b/CONTRIBUTING.md#pull-requests)
requires a human contributor to understand and vouch for changes; fully agentic
PRs are not allowed. Owner review/approval is required before updating that PR.
The tested in-tree patch and Cargo pins remain unchanged meanwhile.
