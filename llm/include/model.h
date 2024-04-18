#ifndef MODEL_H
#define MODEL_H
#include <cstring>

struct model_config {
    int batch;
    int num_heads;
    int num_layers;
    int max_sqlen;
    int embed_dim;
    int hidden_dim;
    int vocsize;
    int padding_idx;
    float rms_norm_eps;  // RMSNorm epsilon (only for LLaMA models)
    // Below are for Clip models
    int image_size;
    int patch_size;
    int projection_dim;
    int mmproj_dim;

    model_config() : model_config(1, 32, 32, 2048, 4096, 11008, 32000, 1, 1e-6, 0, 0, 0, 0) {}
    model_config(int batch, int num_heads, int num_layers, int max_sqlen, int embed_dim, int hidden_dim, int vocsize,
                 int padding_idx, float rms_norm_eps)
        : batch(batch),
          num_heads(num_heads),
          num_layers(num_layers),
          max_sqlen(max_sqlen),
          embed_dim(embed_dim),
          hidden_dim(hidden_dim),
          vocsize(vocsize),
          padding_idx(padding_idx),
          rms_norm_eps(rms_norm_eps) {}
    // Clip models
    model_config(int batch, int num_heads, int num_layers, int max_sqlen, int embed_dim, int hidden_dim, int vocsize,
                 int padding_idx, float rms_norm_eps, int image_size, int patch_size, int projection_dim, int mmproj_dim)
        : batch(batch),
          num_heads(num_heads),
          num_layers(num_layers),
          max_sqlen(max_sqlen),
          embed_dim(embed_dim),
          hidden_dim(hidden_dim),
          vocsize(vocsize),
          padding_idx(padding_idx),
          rms_norm_eps(rms_norm_eps),
          image_size(image_size),
          patch_size(patch_size),
          projection_dim(projection_dim),
          mmproj_dim(mmproj_dim) {}
};

enum { OPT_125M, OPT_1_3B, OPT_6_7B, LLaMA_7B, LLaMA_13B, CodeLLaMA_7B, CodeLLaMA_13B, StarCoder_15_5B, LLaVA_7B, LLaVA_13B, VILA_2_7B, VILA_7B, VILA_13B, Clip_ViT_Large, Mistral_7B};
enum { FP32, QINT8, INT4 };

const struct model_config opt_6_7B(1, 32, 32, 2048, 4096, 16384, 50272, 1, 0);
const struct model_config opt_1_3B(1, 32, 24, 2048, 2048, 8192, 50272, 1, 0);
const struct model_config opt_125m(1, 12, 12, 2048, 768, 3072, 50272, 1, 0);
const struct model_config llama_7B(1, 32, 32, 2048, 4096, 11008, 32000, 1, 1e-6);
const struct model_config llama_13B(1, 40, 40, 2048, 5120, 13824, 32000, 1, 1e-6);
const struct model_config codellama_7B(1, 32, 32, 2048, 4096, 11008, 32016, 1, 1e-5);
const struct model_config codellama_13B(1, 40, 40, 2048, 5120, 13824, 32016, 1, 1e-5);
const struct model_config starcoder_15_5B(1, 48, 40, 2048, 6144, 24576, 49152, 1, 0);
const struct model_config llava_7B(1, 32, 32, 2048, 4096, 11008, 32000, 1, 1e-5);
const struct model_config llava_13B(1, 40, 40, 2048, 5120, 13824, 32000, 1, 1e-5);
const struct model_config vila_2_7B(1, 20, 32, 2048, 2560, 6912, 32000, 1, 1e-5);
const struct model_config vila_7B(1, 32, 32, 2048, 4096, 11008, 32000, 1, 1e-5);
const struct model_config vila_13B(1, 40, 40, 2048, 5120, 13824, 32000, 1, 1e-5);
const struct model_config clip_vit_large(1, 16, 23, 2048, 1024, 4096, 0, 1, 0, 336, 14, 768, 4096); // llava's and vila's clip model uses only 23 layers out of 24
const struct model_config mistral_7B(1, 32, 32, 2048, 4096, 11008, 32000, 1, 1e-6);

static struct model_config get_opt_model_config(int choise) {
    struct model_config ret;
    switch (choise) {
        case OPT_125M:
            ret = opt_125m;
            break;
        case OPT_1_3B:
            ret = opt_1_3B;
            break;
        case OPT_6_7B:
            ret = opt_6_7B;
            break;
        case LLaMA_7B:
            ret = llama_7B;
            break;
        case LLaMA_13B:
            ret = llama_13B;
            break;
        case CodeLLaMA_7B:
            ret = codellama_7B;
            break;
        case CodeLLaMA_13B:
            ret = codellama_13B;
            break;
        case StarCoder_15_5B:
            ret = starcoder_15_5B;
            break;
        case LLaVA_7B:
            ret = llava_7B;
            break;
        case LLaVA_13B:
            ret = llava_13B;
            break;
        case VILA_2_7B:
            ret = vila_2_7B;
            break;
        case VILA_7B:
            ret = vila_7B;
            break;
        case VILA_13B:
            ret = vila_13B;
            break;
        case Clip_ViT_Large:
            ret = clip_vit_large;
            break;
        case Mistral_7B:
            ret = mistral_7B;
            break;
        default:
            throw("Unsupported model choice.");
            break;
    }
    return ret;
}

#endif
