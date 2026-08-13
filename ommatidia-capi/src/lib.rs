//! Stable, panic-free C entry points.
//!
//! The first ABI slice is deliberately limited to version and checkpoint
//! discovery. GPU creation is not exported until Blade can borrow a host
//! Vulkan device and Meganeura can record into host-owned command buffers.

use std::ffi::{CStr, c_char};
use std::mem::size_of;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::Path;

/// ABI 1.0, encoded as `major << 16 | minor`.
pub const API_VERSION: u32 = 1 << 16;

#[repr(i32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Status {
    Ok = 0,
    InvalidArgument = 1,
    IncompatibleStruct = 2,
    Io = 3,
    MalformedConfig = 4,
    UnsupportedModel = 5,
    Internal = 6,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct ModelInfo {
    /// Caller sets this to `sizeof(OmmatidiumModelInfo)`.
    pub struct_size: u32,
    pub api_version: u32,
    pub scale: u32,
    pub training_tile: u32,
    pub extent_alignment: u32,
    pub input_plane_mask: u32,
    pub input_channel_count: u32,
    pub output_channel_count: u32,
    pub objective: u32,
    pub backbone: u32,
    pub attention_window: u32,
    pub attention_head_dim: u32,
    pub parameter_count: u64,
    pub reserved: [u32; 8],
}

impl Default for ModelInfo {
    fn default() -> Self {
        Self {
            struct_size: size_of::<Self>() as u32,
            api_version: API_VERSION,
            scale: 0,
            training_tile: 0,
            extent_alignment: 0,
            input_plane_mask: 0,
            input_channel_count: 0,
            output_channel_count: 0,
            objective: 0,
            backbone: 0,
            attention_window: 0,
            attention_head_dim: 0,
            parameter_count: 0,
            reserved: [0; 8],
        }
    }
}

fn write_error(buffer: *mut c_char, capacity: usize, message: &str) {
    if buffer.is_null() || capacity == 0 {
        return;
    }
    let bytes = message.as_bytes();
    let count = bytes.len().min(capacity - 1);
    // SAFETY: the ABI requires a writable array of `capacity` bytes. We copy
    // at most capacity-1 and always append the terminator.
    unsafe {
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), buffer.cast::<u8>(), count);
        *buffer.add(count) = 0;
    }
}

fn inspect_model(stem: &Path) -> Result<ModelInfo, (Status, String)> {
    let (config, _) = ommatidia::checkpoint::load_config(stem).map_err(|error| match error {
        ommatidia::checkpoint::Error::Io(error) => (Status::Io, error.to_string()),
        ommatidia::checkpoint::Error::Config(message) => (Status::MalformedConfig, message),
    })?;
    config
        .validate()
        .map_err(|message| (Status::UnsupportedModel, message))?;
    let model = ommatidia::model::build(&config, false)
        .map_err(|message| (Status::UnsupportedModel, message))?;
    let (backbone, attention_window, attention_head_dim) = match config.backbone {
        ommatidia::Backbone::Conv => (1, 0, 0),
        ommatidia::Backbone::HybridWindowAttention { window, head_dim } => (2, window, head_dim),
    };
    let objective = match config.objective {
        ommatidia::Objective::Direct => 1,
        ommatidia::Objective::Diffusion => 2,
    };
    Ok(ModelInfo {
        scale: config.scale,
        training_tile: config.tile,
        extent_alignment: 1 << (config.levels() - 1),
        input_plane_mask: config.cond_planes.bits(),
        input_channel_count: config.cond_channels(),
        output_channel_count: config.target_channels(),
        objective,
        backbone,
        attention_window,
        attention_head_dim,
        parameter_count: model
            .params
            .iter()
            .map(|parameter| parameter.len as u64)
            .sum(),
        ..ModelInfo::default()
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn ommatidia_api_version() -> u32 {
    API_VERSION
}

#[unsafe(no_mangle)]
pub extern "C" fn ommatidia_status_string(status: i32) -> *const c_char {
    match status {
        0 => c"ok".as_ptr(),
        1 => c"invalid argument".as_ptr(),
        2 => c"incompatible structure size".as_ptr(),
        3 => c"I/O error".as_ptr(),
        4 => c"malformed checkpoint config".as_ptr(),
        5 => c"unsupported model".as_ptr(),
        _ => c"internal error".as_ptr(),
    }
}

/// Inspect a checkpoint sidecar without creating or enumerating a GPU.
///
/// `checkpoint_stem` may include `.ron`; it is treated the same as a bare
/// stem. `out_info->struct_size` must be initialized by the caller.
///
/// # Safety
///
/// `checkpoint_stem` must point to a readable NUL-terminated string.
/// `out_info` must be aligned and writable for at least the number of bytes in
/// its `struct_size` field. When non-null, `error_message` must be writable for
/// `error_message_capacity` bytes. All pointers need only remain valid for the
/// duration of this call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ommatidia_model_inspect(
    checkpoint_stem: *const c_char,
    out_info: *mut ModelInfo,
    error_message: *mut c_char,
    error_message_capacity: usize,
) -> Status {
    if checkpoint_stem.is_null() || out_info.is_null() {
        write_error(
            error_message,
            error_message_capacity,
            "checkpoint_stem and out_info are required",
        );
        return Status::InvalidArgument;
    }
    // SAFETY: pointers were checked for null; validity remains the caller's
    // documented ABI obligation.
    let caller_size = unsafe { (*out_info).struct_size } as usize;
    if caller_size < size_of::<ModelInfo>() {
        write_error(
            error_message,
            error_message_capacity,
            "OmmatidiumModelInfo is smaller than ABI 1.0 requires",
        );
        return Status::IncompatibleStruct;
    }
    let result = catch_unwind(AssertUnwindSafe(|| {
        // SAFETY: the ABI requires a NUL-terminated string.
        let stem = unsafe { CStr::from_ptr(checkpoint_stem) }
            .to_str()
            .map_err(|_| {
                (
                    Status::InvalidArgument,
                    "checkpoint path is not valid UTF-8".to_string(),
                )
            })?;
        inspect_model(Path::new(stem))
    }));
    match result {
        Ok(Ok(info)) => {
            // SAFETY: size validation above guarantees the known ABI prefix is
            // writable. A future ABI can preserve smaller historical prefixes.
            unsafe { out_info.write(info) };
            write_error(error_message, error_message_capacity, "");
            Status::Ok
        }
        Ok(Err((status, message))) => {
            write_error(error_message, error_message_capacity, &message);
            status
        }
        Err(_) => {
            write_error(
                error_message,
                error_message_capacity,
                "internal panic while inspecting the model",
            );
            Status::Internal
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::CString;

    #[test]
    fn api_version_is_major_one() {
        assert_eq!(ommatidia_api_version() >> 16, 1);
        assert_eq!(unsafe { CStr::from_ptr(ommatidia_status_string(0)) }, c"ok");
    }

    #[test]
    fn inspection_reports_the_graph_contract() {
        let dir = std::env::temp_dir().join("ommatidia-capi-inspect");
        std::fs::create_dir_all(&dir).unwrap();
        let stem = dir.join("model");
        let config = ommatidia::ModelConfig::default();
        std::fs::write(
            stem.with_extension("ron"),
            ron::ser::to_string(&config).unwrap(),
        )
        .unwrap();
        let stem = CString::new(stem.to_str().unwrap()).unwrap();
        let mut info = ModelInfo::default();
        let mut error = [0 as c_char; 256];
        let status = unsafe {
            ommatidia_model_inspect(stem.as_ptr(), &mut info, error.as_mut_ptr(), error.len())
        };
        assert_eq!(status, Status::Ok, "{:?}", unsafe {
            CStr::from_ptr(error.as_ptr())
        });
        assert_eq!(info.scale, 2);
        assert_eq!(info.input_plane_mask, config.cond_planes.bits());
        assert_eq!(info.backbone, 1);
        assert_eq!(info.parameter_count, 649_200);
        std::fs::remove_file(stem.to_str().unwrap().to_owned() + ".ron").unwrap();
    }

    #[test]
    fn too_small_output_is_rejected_without_writing_it() {
        let stem = c"unused";
        let mut info = ModelInfo {
            struct_size: 4,
            ..ModelInfo::default()
        };
        let status =
            unsafe { ommatidia_model_inspect(stem.as_ptr(), &mut info, std::ptr::null_mut(), 0) };
        assert_eq!(status, Status::IncompatibleStruct);
        assert_eq!(info.struct_size, 4);
    }
}
