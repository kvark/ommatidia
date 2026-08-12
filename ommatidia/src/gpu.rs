//! Context helpers for standalone tools and tests.
//!
//! Host integrations do not use these to choose a device: [`crate::Upscaler`]
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
/// [`crate::Upscaler`] instead. Timing has to be decided before context
/// creation because Blade allocates its timestamp query pools up front.
pub fn create_context(device_id: Option<u32>, timing: bool) -> Arc<blade_graphics::Context> {
    let context = meganeura::init_gpu_context_with(meganeura::GpuOptions { device_id, timing })
        .expect("failed to initialise a GPU context");
    log::info!("using {}", context.device_information().device_name);
    Arc::new(context)
}

/// What else is using the device right now.
///
/// Worth checking before believing a timing. A busy device compresses every
/// ratio toward one; a nearly-full one does something worse, because an
/// allocation that no longer fits is served over PCIe instead, and that
/// penalises whichever path moves the most data — which is exactly the path a
/// comparison is usually trying to judge. A Winograd-versus-direct comparison
/// on this machine came out 1.4x one way on an idle device and 2.4x the other
/// way with 19 of 20 GiB spoken for.
#[derive(Clone, Copy, Debug)]
pub struct DeviceLoad {
    pub busy_percent: u32,
    pub vram_used_mib: u64,
    pub vram_total_mib: u64,
}

impl DeviceLoad {
    /// Read the amdgpu sysfs counters, if they are there.
    ///
    /// `None` on any other driver or platform — this is a diagnostic, so it
    /// declines to guess rather than reporting something made up.
    pub fn read() -> Option<Self> {
        let base = std::path::Path::new("/sys/class/drm/card0/device");
        let read = |name: &str| -> Option<u64> {
            std::fs::read_to_string(base.join(name))
                .ok()?
                .trim()
                .parse()
                .ok()
        };
        Some(Self {
            busy_percent: read("gpu_busy_percent")? as u32,
            vram_used_mib: read("mem_info_vram_used")? / 1024 / 1024,
            vram_total_mib: read("mem_info_vram_total")? / 1024 / 1024,
        })
    }

    pub fn vram_free_mib(&self) -> u64 {
        self.vram_total_mib.saturating_sub(self.vram_used_mib)
    }

    /// Whether a timing taken now is worth trusting in absolute terms.
    pub fn is_quiet(&self) -> bool {
        self.busy_percent < 20 && self.vram_free_mib() > 4096
    }
}

/// Print what the device is doing, and say plainly when a measurement taken
/// now should not be believed.
pub fn warn_if_busy() {
    let Some(load) = DeviceLoad::read() else {
        return;
    };
    if load.is_quiet() {
        println!(
            "device idle: {}% busy, {} MiB VRAM free",
            load.busy_percent,
            load.vram_free_mib()
        );
        return;
    }
    println!(
        "!! device is busy: {}% used, {} of {} MiB VRAM free.\n\
         !! Absolute timings are inflated, and ratios between paths that move \
         different amounts of memory can invert outright. Re-measure idle.",
        load.busy_percent,
        load.vram_free_mib(),
        load.vram_total_mib
    );
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
