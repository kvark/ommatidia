//! Choosing the device, in one place.
//!
//! Meganeura's core is environment-free: it takes typed options and never
//! reads `MEGANEURA_*` itself, so a client that wants a specific adapter has
//! to say so. Ommatidia says so here, through its own `OMMATIDIA_DEVICE_ID`,
//! which also covers the generator — that builds a Blade context directly and
//! never goes through meganeura at all.
//!
//! This matters more than a convenience wrapper usually would. A session built
//! with no options lands on whichever adapter the driver enumerates first,
//! which on a machine with an integrated and a discrete GPU is the integrated
//! one. Nothing fails; the numbers are just quietly wrong.

use std::sync::Arc;

/// Environment variable naming the adapter to run on.
pub const DEVICE_ID_VAR: &str = "OMMATIDIA_DEVICE_ID";

/// Adapter selection, by the backend-reported numeric device ID.
///
/// On Vulkan that is the PCI device ID rather than an adapter ordinal, so it
/// is conventionally written in hex; decimal is accepted too. `None` leaves
/// the choice to the driver.
pub fn device_id() -> Option<u32> {
    let value = std::env::var(DEVICE_ID_VAR).ok()?;
    let value = value.trim();
    let parsed = match value
        .strip_prefix("0x")
        .or_else(|| value.strip_prefix("0X"))
    {
        Some(hex) => u32::from_str_radix(hex, 16).ok(),
        None => value.parse().ok(),
    };
    if parsed.is_none() {
        log::warn!("ignoring invalid {DEVICE_ID_VAR}={value:?}");
    }
    parsed
}

/// GPU options for a meganeura session: the selected adapter, and whether to
/// allocate timestamp query pools.
///
/// Timing has to be decided before the context exists, which is why it is
/// here rather than on the session.
pub fn options(timing: bool) -> meganeura::GpuOptions {
    meganeura::GpuOptions {
        device_id: device_id(),
        timing,
    }
}

/// A meganeura GPU context on the selected adapter.
///
/// Share one of these across every session in a process: a context is not
/// cheap, and two of them on one device contend for the same queue.
pub fn context(timing: bool) -> Arc<blade_graphics::Context> {
    let context = meganeura::init_gpu_context_with(options(timing))
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

    /// Guarded so a developer's own setting cannot make this pass or fail.
    #[test]
    fn device_id_accepts_hex_and_decimal() {
        // Safety: single-threaded within this test, and restored before it
        // returns. `--test-threads=1` is not required because no other test
        // in this crate reads the variable.
        let restore = std::env::var(DEVICE_ID_VAR).ok();
        unsafe {
            std::env::set_var(DEVICE_ID_VAR, "0x744c");
            assert_eq!(device_id(), Some(0x744c));
            std::env::set_var(DEVICE_ID_VAR, "  0X744C  ");
            assert_eq!(device_id(), Some(0x744c));
            std::env::set_var(DEVICE_ID_VAR, "29772");
            assert_eq!(device_id(), Some(29772));
            // Unparsable warns and falls back rather than panicking.
            std::env::set_var(DEVICE_ID_VAR, "not-a-device");
            assert_eq!(device_id(), None);
            std::env::remove_var(DEVICE_ID_VAR);
            assert_eq!(device_id(), None);
            match restore {
                Some(v) => std::env::set_var(DEVICE_ID_VAR, v),
                None => std::env::remove_var(DEVICE_ID_VAR),
            }
        }
    }
}
