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
    int qk;  // group size

    model_config() : model_config(1, 12, 12, 512, 768, 3072, 50272, 1) {}
    model_config(int batch, int num_heads, int num_layers, int max_sqlen, int embed_dim, int hidden_dim, int vocsize,
                 int padding_idx)
        : batch(batch),
          num_heads(num_heads),
          num_layers(num_layers),
          max_sqlen(max_sqlen),
          embed_dim(embed_dim),
          hidden_dim(hidden_dim),
          vocsize(vocsize),
          padding_idx(padding_idx) {}
};

enum { OPT_125M, OPT_1_3B, OPT_6_7B, LLaMA_7B };
enum { FP32, INT8, INT4 };

const struct model_config opt_6_7B(1, 32, 32, 2048, 4096, 16384, 50272, 1);
const struct model_config opt_1_3B(1, 32, 24, 2048, 2048, 8192, 50272, 1);
const struct model_config opt_125m(1, 12, 12, 2048, 768, 3072, 50272, 1);
const struct model_config llama_7B(1, 32, 32, 2048, 4096, 11008, 32000, 1);
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
        case LLaMA_7B:;
            ret = llama_7B;
            break;
        default:
            throw("Unsupported model choise.");
            break;
    }
    return ret;
}

#endif
