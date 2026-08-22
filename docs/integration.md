# Native integration and releases

Ommatidium should be a user-space library, not a Vulkan extension. Vulkan
extensions describe implementation capabilities exposed by a driver or layer;
this project is application middleware with model weights, mutable history,
and a release cadence independent of a Vulkan implementation.

The target is a versioned C ABI with thin Rust and C++ conveniences. A host
will provide its already-selected device/queue, input image views, formats and
layouts, synchronization points, frame constants, and an output image. The
library must never enumerate adapters or create a second device on this path.
Standalone training, capture, and benchmark binaries may still accept an
explicit adapter ID because they are the application in that context.

The proposed Vulkan entry point imports, but does not own:

- `VkInstance`, `VkPhysicalDevice`, `VkDevice`, `VkQueue`, and queue-family
  index;
- dispatch loading through the host's `vkGetInstanceProcAddr`;
- `VkImageView` inputs and output plus their declared layouts;
- a command buffer supplied by the host, or timeline semaphore values when the
  library submits internally.

Recording into the host command buffer is the preferred end state: it avoids
hidden queue submissions and makes barriers and profiler scopes visible to the
application. Meganeura currently owns and submits its encoder, while
blade-graphics can create a device but cannot wrap externally owned Vulkan
handles. Those are the two prerequisites before a truthful C *inference*
example can be shipped.

This should begin as an ordinary exported API over core Vulkan handles. A
private Vulkan extension would require a layer or driver implementation and
still would not solve model discovery, versioning, history ownership, or
application synchronization. Standard interop extensions such as external
memory and timeline semaphores remain useful implementation tools when direct
command-buffer recording is unavailable; they are not the product surface.

The ABI will use opaque handles, fixed-width integers, explicit structure
sizes/version fields, status returns, and caller-provided logging callbacks.
No Rust types, exceptions, allocator ownership, environment variables, or
process-global device selection cross the boundary.

Checkpoint inspection also exposes whether reconstruction needs
output-resolution depth, world normal, and diffuse albedo. These are borrowed
read-only image views just like the low-resolution inputs. A raster/deferred
host may already own them; a pure low-resolution ray tracer may need a separate
primary-surface pass. That pass is deliberately host-owned and visible to its
profiler rather than hidden inside Ommatidium. Checkpoints using the older
low-resolution guide require no output-resolution planes.

The Rust API already exercises the intended ownership model when the host uses
blade-graphics: `Upscaler` receives the host's existing `Arc<Context>` and
`FrameInputs` borrows its texture views. Temporal checkpoints additionally
borrow current-to-previous motion, retain their own ping-ponged history, and
provide `reset_history` for cuts. They do not enumerate an adapter or create a
second device. What remains below is specifically the C/external-Vulkan path,
where blade-graphics cannot yet wrap handles created by another graphics
stack.

## Release shape

A tagged GitHub release should eventually contain one archive per supported
platform with:

- the shared library and C header;
- a minimal C sample and its build files;
- the default model sidecar and safetensors file;
- licenses, checksums, and a machine-readable manifest naming ABI, model, and
  required GPU features;
- a small conformance scene whose output is checked by the same image reference
  test as CI.

GitHub remains the source and binary-release authority. Hugging Face remains
the stable model and dataset registry; release packaging pins an immutable HF
revision and verifies its checksum rather than creating architecture-specific
repository names. The GitHub release and HF model card cross-link the same
semantic version.

The `ommatidia-capi` crate now establishes ABI 1.1 for version negotiation,
panic-free status/error handling, and checkpoint discovery. Its
[`examples/c/inspect.c`](../examples/c/inspect.c) consumer compiles and links
as C, and can query the model's scale, planes, alignment, backbone, and
parameter count without touching a GPU. ABI 1.1 also reports the deterministic
reconstruction mode and the exact high-resolution plane mask it requires, so a
host can reject an incompatible checkpoint before allocating frame resources.
This is useful release groundwork,
but it is deliberately not presented as native inference. The public header
states that Vulkan execution is unavailable until the borrowed-device and
host-command-buffer prerequisites below land.

## Groundwork sequence

1. Teach blade-graphics to borrow an externally owned Vulkan device and queue
   without enumerating adapters or destroying host handles.
2. Let Meganeura record a prepared session into a caller-provided encoder or
   command buffer, with no implicit submission.
3. Extend the existing C ABI and C conformance example with the resource and
   synchronization contract. Do not add a device-ID field: creation borrows
   the handles the host supplies.
4. Add release archives and provenance only after that example builds and runs
   against an installed archive rather than the Rust workspace.
