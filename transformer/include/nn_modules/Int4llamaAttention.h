#include <utility>

#include "common.h"
#include "operators.h"

struct Int4llamaAttention_output {
#ifdef USE_CUDA
    Matrix3D<float16_t> attn_output;
    Matrix3D<float16_t> attn_probs_reshaped;
    std::pair<Matrix3D<float16_t>, Matrix3D<float16_t>> past_key_value;
#else
    Matrix3D<float> attn_output;
    Matrix3D<float> attn_probs_reshaped;
    std::pair<Matrix3D<float>, Matrix3D<float>> past_key_value;
#endif
};
struct Int4llamaAttention_input {
    bool has_past_key_value = false;
    int layer_idx;
#ifdef USE_CUDA
    Matrix3D<float16_t> hidden_states;
    Matrix3D<float16_t> attention_mask;
    Matrix3D<float16_t> past_key, past_value;

    Int4llamaAttention_input(Matrix3D<float16_t> hidden_states_, Matrix3D<float16_t> attention_mask_, int layer_idx_)
#else
    Matrix3D<float> hidden_states;
    Matrix3D<float> attention_mask;
    Matrix3D<float> past_key, past_value;

    Int4llamaAttention_input(Matrix3D<float> hidden_states_, Matrix3D<float> attention_mask_, int layer_idx_)
#endif
        : hidden_states(hidden_states_), attention_mask(attention_mask_), layer_idx(layer_idx_) {}

#ifdef USE_CUDA
    Int4llamaAttention_input(Matrix3D<float16_t> hidden_states_, Matrix3D<float16_t> attention_mask_, Matrix3D<float16_t> past_key_,
                             Matrix3D<float16_t> past_value_, bool has_past_key_value_, int layer_idx_)
#else
    Int4llamaAttention_input(Matrix3D<float> hidden_states_, Matrix3D<float> attention_mask_, Matrix3D<float> past_key_,
                             Matrix3D<float> past_value_, bool has_past_key_value_, int layer_idx_)
#endif
        : hidden_states(hidden_states_),
          attention_mask(attention_mask_),
          past_key(past_key_),
          past_value(past_value_),
          has_past_key_value(has_past_key_value_),
          layer_idx(layer_idx_) {}
};

class Int4llamaAttention {
   public:
    Int4llamaAttention(std::string param_path, const struct model_config config);
    Int4llamaAttention() {}
    static void initialized_memory(const struct model_config config);
    // void initialized_memory(const struct model_config config);
    void free_cuda_memory();
    struct Int4llamaAttention_output forward(const struct Int4llamaAttention_input &input);
// #ifdef USE_CUDA
//     float16_t *attn_weights_arr;
//     float16_t *attn_output_half_arr;
//     float16_t *query_states_unshape_arr;
//     float16_t *attn_output_arr;
//     float16_t *attn_output_transpose_arr;
//     float16_t *key_states_unshape_arr;
//     float16_t *key_states_arr;
//     float16_t *value_states_unshape_arr;
//     float16_t *value_states_arr;
//     float16_t *query_states_arr;
//     float16_t *value_states_transpose_arr;
//     float16_t *key_states_arr_cache;
//     float16_t *value_states_arr_cache;
//     // int *cache_num;

//     // Linear_half_int4 k_proj, v_proj, q_proj, o_proj;
//     // RotaryPosEmb_cuda rotary_pos_emb;
//     // BMM_F16T qk_bmm, pv_bmm;
//     // int max_sqlen;
// // #else
//     // Linear_FP_int4 k_proj, v_proj, q_proj, o_proj;
//     // RotaryPosEmb rotary_pos_emb;
//     // BMM_F32T qk_bmm, pv_bmm;
//     // void unshape(Matrix3D<float> shaped, Matrix3D<float> unshape, int sqlen);
//     // void shape(Matrix3D<float> unshape, Matrix3D<float> shaped, int sqlen);
// #endif
//     // std::string profile_name = "Int4llamaAttention";
//     // int embed_dim, num_heads, head_dim;

    int *q_weight = nullptr, *k_weight = nullptr, *v_weight = nullptr, *o_weight = nullptr;
    half *cos_buf = nullptr, *sin_buf = nullptr;

   private:
    std::string profile_name = "Int4llamaAttention";
    int embed_dim, num_heads, head_dim;
#ifdef USE_CUDA
    Linear_half_int4 k_proj, v_proj, q_proj, o_proj;
    RotaryPosEmb_cuda rotary_pos_emb;
    BMM_F16T qk_bmm, pv_bmm;
    int max_sqlen;
#else
    Linear_FP_int4 k_proj, v_proj, q_proj, o_proj;
    RotaryPosEmb rotary_pos_emb;
    BMM_F32T qk_bmm, pv_bmm;
    void unshape(Matrix3D<float> shaped, Matrix3D<float> unshape, int sqlen);
    void shape(Matrix3D<float> unshape, Matrix3D<float> shaped, int sqlen);
#endif
};
