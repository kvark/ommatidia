#ifndef OMMATIDIA_H
#define OMMATIDIA_H

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
#  if defined(OMMATIDIA_BUILD_SHARED)
#    define OMMATIDIA_API __declspec(dllexport)
#  elif defined(OMMATIDIA_USE_SHARED)
#    define OMMATIDIA_API __declspec(dllimport)
#  else
#    define OMMATIDIA_API
#  endif
#else
#  define OMMATIDIA_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define OMMATIDIA_API_VERSION_MAJOR 1u
#define OMMATIDIA_API_VERSION_MINOR 0u
#define OMMATIDIA_MAKE_VERSION(major, minor) (((major) << 16u) | (minor))
#define OMMATIDIA_API_VERSION \
    OMMATIDIA_MAKE_VERSION(OMMATIDIA_API_VERSION_MAJOR, OMMATIDIA_API_VERSION_MINOR)

typedef enum OmmatidiumStatus {
    OMMATIDIA_STATUS_OK = 0,
    OMMATIDIA_STATUS_INVALID_ARGUMENT = 1,
    OMMATIDIA_STATUS_INCOMPATIBLE_STRUCT = 2,
    OMMATIDIA_STATUS_IO = 3,
    OMMATIDIA_STATUS_MALFORMED_CONFIG = 4,
    OMMATIDIA_STATUS_UNSUPPORTED_MODEL = 5,
    OMMATIDIA_STATUS_INTERNAL = 6
} OmmatidiumStatus;

typedef enum OmmatidiumObjective {
    OMMATIDIA_OBJECTIVE_DIRECT = 1,
    OMMATIDIA_OBJECTIVE_DIFFUSION = 2
} OmmatidiumObjective;

typedef enum OmmatidiumBackbone {
    OMMATIDIA_BACKBONE_CONV = 1,
    OMMATIDIA_BACKBONE_HYBRID_WINDOW_ATTENTION = 2
} OmmatidiumBackbone;

typedef struct OmmatidiumModelInfo {
    /* Set to sizeof(OmmatidiumModelInfo) before calling. */
    uint32_t struct_size;
    uint32_t api_version;
    uint32_t scale;
    uint32_t training_tile;
    uint32_t extent_alignment;
    uint32_t input_plane_mask;
    uint32_t input_channel_count;
    uint32_t output_channel_count;
    uint32_t objective;
    uint32_t backbone;
    uint32_t attention_window;
    uint32_t attention_head_dim;
    uint64_t parameter_count;
    uint32_t reserved[8];
} OmmatidiumModelInfo;

OMMATIDIA_API uint32_t ommatidia_api_version(void);
OMMATIDIA_API const char *ommatidia_status_string(int32_t status);

/*
 * Inspect <checkpoint_stem>.ron without creating or enumerating a GPU.
 * error_message may be NULL when error_message_capacity is zero.
 */
OMMATIDIA_API OmmatidiumStatus ommatidia_model_inspect(
    const char *checkpoint_stem,
    OmmatidiumModelInfo *out_info,
    char *error_message,
    size_t error_message_capacity);

/*
 * Vulkan execution is intentionally not exposed in ABI 1.0 yet. The native
 * path will borrow the host-selected VkDevice/queue and record into a supplied
 * command buffer; it will not accept a device ID or enumerate adapters.
 */

#ifdef __cplusplus
}
#endif

#endif /* OMMATIDIA_H */
