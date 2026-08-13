# C ABI example

ABI 1.1 proves distribution, linking, error handling, and checkpoint metadata
without pretending that a second internally-created Vulkan device is native
integration. Build the library and example with:

```sh
cargo build --release -p ommatidia-capi
cmake -S examples/c -B target/c-example
cmake --build target/c-example
LD_LIBRARY_PATH=target/release target/c-example/ommatidia-inspect
```

With no checkpoint argument the executable only verifies that the header and
shared library agree on the ABI version. Pass `/path/to/checkpoint-stem` (with
or without `.ron`) to print that model's contract. GPU execution will be added
after Blade can wrap borrowed Vulkan handles and Meganeura can record into a
host-provided command buffer. That API will take the already-selected device
and queue; there will be no device-ID field.
