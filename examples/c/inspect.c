#include "ommatidia.h"

#include <inttypes.h>
#include <stdio.h>
#include <string.h>

int main(int argc, char **argv) {
    const uint32_t version = ommatidia_api_version();
    printf("Ommatidia C ABI %u.%u\n", version >> 16u, version & 0xffffu);
    if (version != OMMATIDIA_API_VERSION) {
        fprintf(stderr, "header/library ABI mismatch\n");
        return 2;
    }
    if (argc == 1) {
        printf("pass a checkpoint stem to inspect its graph contract\n");
        return 0;
    }

    OmmatidiumModelInfo info;
    memset(&info, 0, sizeof(info));
    info.struct_size = (uint32_t)sizeof(info);
    char error[512] = {0};
    const OmmatidiumStatus status =
        ommatidia_model_inspect(argv[1], &info, error, sizeof(error));
    if (status != OMMATIDIA_STATUS_OK) {
        fprintf(stderr, "inspect failed: %s", ommatidia_status_string(status));
        if (error[0] != '\0') {
            fprintf(stderr, ": %s", error);
        }
        fputc('\n', stderr);
        return 1;
    }

    printf("scale: %ux\n", info.scale);
    printf("training tile: %u, runtime alignment: %u\n",
           info.training_tile, info.extent_alignment);
    printf("planes: 0x%08x, channels: %u -> %u\n",
           info.input_plane_mask, info.input_channel_count,
           info.output_channel_count);
    printf("objective: %u, backbone: %u, parameters: %" PRIu64 "\n",
           info.objective, info.backbone, info.parameter_count);
    printf("reconstruction base: %u, required HR planes: 0x%08x\n",
           info.reconstruction_base, info.required_hr_plane_mask);
    if (info.backbone == OMMATIDIA_BACKBONE_HYBRID_WINDOW_ATTENTION) {
        printf("attention: %ux%u windows, head dimension %u\n",
               info.attention_window, info.attention_window,
               info.attention_head_dim);
    }
    return 0;
}
