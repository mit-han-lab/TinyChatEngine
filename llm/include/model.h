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

    model_config() : model_config(1, 32, 32, 2048, 4096, 11008, 32000, 1, 1e-6) {}
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
};

enum { OPT_125M, OPT_1_3B, OPT_6_7B, LLaMA_7B, LLaMA_13B, CodeLLaMA_7B, CodeLLaMA_13B, StarCoder_15_5B };
enum { FP32, QINT8, INT4 };

const struct model_config opt_6_7B(1, 32, 32, 2048, 4096, 16384, 50272, 1, 0);
const struct model_config opt_1_3B(1, 32, 24, 2048, 2048, 8192, 50272, 1, 0);
const struct model_config opt_125m(1, 12, 12, 2048, 768, 3072, 50272, 1, 0);
const struct model_config llama_7B(1, 32, 32, 2048, 4096, 11008, 32000, 1, 1e-6);
const struct model_config llama_13B(1, 40, 40, 2048, 5120, 13824, 32000, 1, 1e-6);
const struct model_config codellama_7B(1, 32, 32, 2048, 4096, 11008, 32016, 1, 1e-5);
const struct model_config codellama_13B(1, 40, 40, 2048, 5120, 13824, 32016, 1, 1e-5);
// const struct model_config starcoder_15_5B(1, 32, 32, 2048, 4096, 11008, 32000, 1, 0); // temporary
const struct model_config starcoder_15_5B(1, 48, 40, 2048, 6144, 24576, 49152, 1, 0);
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
        default:
            throw("Unsupported model choice.");
            break;
    }
    return ret;
}

#endif
