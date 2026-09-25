//! Context helpers for standalone tools and tests.
//!
//! Host integrations do not use these to choose a device: [`crate::transport::native::Native`]
//! takes the host's existing `Arc<blade_graphics::Context>`. The trainer and
//! GPU tests have no host, so they create a context from explicit typed
//! arguments at their CLI/test boundary.

use std::sync::Arc;

/// Adapter selection, by the backend-reported numeric device ID.
///
/// On Vulkan that is the PCI device ID rather than an adapter ordinal, so it
/// is conventionally written in hex; decimal is accepted too.
pub fn parse_device_id(value: &str) -> Result<u32, String> {
    let value = value.trim();
    match value
        .strip_prefix("0x")
        .or_else(|| value.strip_prefix("0X"))
    {
        Some(hex) => u32::from_str_radix(hex, 16),
        None => value.parse(),
    }
    .map_err(|error| format!("invalid device id {value:?}: {error}"))
}

/// Create a Meganeura GPU context for a standalone caller.
///
/// Integrated runtimes should pass their existing Blade context to
/// [`crate::transport::native::Native`] instead. Timing has to be decided before context
/// creation because Blade allocates its timestamp query pools up front.
pub fn create_context(device_id: Option<u32>, timing: bool) -> Arc<blade_graphics::Context> {
    let context = meganeura::init_gpu_context_with(meganeura::GpuOptions {
        device_id,
        timing,
        ..Default::default()
    })
    .expect("failed to initialise a GPU context");
    log::info!("using {}", context.device_information().device_name);
    Arc::new(context)
}

/// Build an inference session for `graph` on the selected adapter.
///
/// The bare `meganeura::build_inference_session` takes no options and so
/// lands wherever the driver puts it; this is the same call with the device
/// choice supplied.
pub fn inference_session(
    graph: &meganeura::Graph,
    context: Arc<blade_graphics::Context>,
) -> meganeura::Session {
    meganeura::train::build(
        graph,
        meganeura::SessionConfig {
            mode: meganeura::Mode::Inference,
            gpu: Some(context),
            ..Default::default()
        },
    )
    .0
}

/// Build a training session for `graph` on the selected adapter.
pub fn training_session(
    graph: &meganeura::Graph,
    context: Arc<blade_graphics::Context>,
) -> meganeura::Session {
    meganeura::train::build(
        graph,
        meganeura::SessionConfig {
            mode: meganeura::Mode::Training,
            gpu: Some(context),
            ..Default::default()
        },
    )
    .0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn device_id_accepts_hex_and_decimal() {
        assert_eq!(parse_device_id("0x744c"), Ok(0x744c));
        assert_eq!(parse_device_id("  0X744C  "), Ok(0x744c));
        assert_eq!(parse_device_id("29772"), Ok(29772));
        assert!(parse_device_id("not-a-device").is_err());
    }
}
