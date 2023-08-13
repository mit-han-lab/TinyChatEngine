#include "Int4llamaAttention.h"
#include "common.h"
#include "operators.h"

struct Int4llamaDecoderLayer_output {
#ifdef USE_CUDA
    Matrix3D<float16_t> hidden_states;
    Matrix3D<float16_t> attentions;
    std::pair<Matrix3D<float16_t>, Matrix3D<float16_t>> past_key_value;

    Int4llamaDecoderLayer_output(Matrix3D<float16_t> hidden_states_, Matrix3D<float16_t> attentions_,
                                 std::pair<Matrix3D<float16_t>, Matrix3D<float16_t>> past_key_value_) {
#else
    Matrix3D<float> hidden_states;
    Matrix3D<float> attentions;
    std::pair<Matrix3D<float>, Matrix3D<float>> past_key_value;

    Int4llamaDecoderLayer_output(Matrix3D<float> hidden_states_, Matrix3D<float> attentions_,
                                 std::pair<Matrix3D<float>, Matrix3D<float>> past_key_value_) {
#endif
        hidden_states = hidden_states_;
        attentions = attentions_;
        past_key_value = past_key_value_;
    };
};
struct Int4llamaDecoderLayer_input {
    bool has_past_key_value = false;
#ifdef USE_CUDA
    Matrix3D<float16_t> hidden_states;
    Matrix3D<float16_t> attention_mask;
    Matrix3D<float16_t> past_key, past_value;

    Int4llamaDecoderLayer_input(Matrix3D<float16_t> hidden_states_, Matrix3D<float16_t> attention_mask_) {
#else
    Matrix3D<float> hidden_states;
    Matrix3D<float> attention_mask;
    Matrix3D<float> past_key, past_value;

    Int4llamaDecoderLayer_input(Matrix3D<float> &hidden_states_, Matrix3D<float> &attention_mask_) {
#endif
        hidden_states = hidden_states_;
        attention_mask = attention_mask_;
        has_past_key_value = false;
    }

#ifdef USE_CUDA
    Int4llamaDecoderLayer_input(Matrix3D<float16_t> hidden_states_, Matrix3D<float16_t> attention_mask_,
                                Matrix3D<float16_t> past_key_, Matrix3D<float16_t> past_value_) {
#else
    Int4llamaDecoderLayer_input(Matrix3D<float> &hidden_states_, Matrix3D<float> &attention_mask_,
                                Matrix3D<float> past_key_, Matrix3D<float> past_value_) {
#endif
        hidden_states = hidden_states_;
        attention_mask = attention_mask_;
        past_key = past_key_;
        past_value = past_value_;
        has_past_key_value = true;
    }
};

class Int4llamaDecoderLayer {
   public:
    Int4llamaDecoderLayer(std::string param_path, const struct model_config config, int layer_idx);
    Int4llamaDecoderLayer() {};
    void free_cuda_memory();
    struct Int4llamaDecoderLayer_output forward(const struct Int4llamaDecoderLayer_input &input);

    std::string profile_name = "Int4llamaDecoderLayer";
    int embed_dim, num_attention_heads, hidden_dim, layer_idx;
    Int4llamaAttention attn;
#ifdef USE_CUDA
    LlamaRMSNorm_cuda input_layernorm, post_attention_layernorm;
    Linear_half_int4 gate_proj, down_proj, up_proj;
    // float16_t* split_8_buffer;
    int *gate_proj_weight = nullptr, *down_proj_weight = nullptr, *up_proj_weight = nullptr;

    // float16_t *hidden_states_half_arr;
    // float16_t *final_layer_norm_arr;
    // float16_t *gate_proj_arr;
    // float16_t *up_proj_arr;
    // float16_t *down_proj_arr;
    // float16_t *hidden_states_arr;
#else
    LlamaRMSNorm input_layernorm, post_attention_layernorm;  // from torch_int.nn
    Linear_FP_int4 gate_proj, down_proj, up_proj;
#endif
    float *input_layernorm_weight_ptr = nullptr;
    float *post_attention_layernorm_ptr = nullptr;
};
