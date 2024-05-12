#include <utility>

#include "common.h"
#include "operators.h"

struct Int4llamaAttention_output {
#ifdef QM_CUDA
    Matrix3D<float16_t> attn_output;
    Matrix3D<float16_t> attn_probs_reshaped;
    std::pair<Matrix3D<float16_t>, Matrix3D<float16_t>> past_key_value;
#elif defined(QM_METAL)
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
#ifdef QM_CUDA
    Matrix3D<float16_t> hidden_states;
    Matrix3D<float16_t> attention_mask;
    Matrix3D<float16_t> past_key, past_value;

    Int4llamaAttention_input(Matrix3D<float16_t> hidden_states_, Matrix3D<float16_t> attention_mask_, int layer_idx_)
#elif defined(QM_METAL)
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
        : hidden_states(hidden_states_), attention_mask(attention_mask_), layer_idx(layer_idx_) {
    }

#ifdef QM_CUDA
    Int4llamaAttention_input(Matrix3D<float16_t> hidden_states_, Matrix3D<float16_t> attention_mask_,
                             Matrix3D<float16_t> past_key_, Matrix3D<float16_t> past_value_, bool has_past_key_value_,
                             int layer_idx_)
#elif defined(QM_METAL)
    Int4llamaAttention_input(Matrix3D<float16_t> hidden_states_, Matrix3D<float16_t> attention_mask_,
                             Matrix3D<float16_t> past_key_, Matrix3D<float16_t> past_value_, bool has_past_key_value_,
                             int layer_idx_)
#else
    Int4llamaAttention_input(Matrix3D<float> hidden_states_, Matrix3D<float> attention_mask_, Matrix3D<float> past_key_,
                             Matrix3D<float> past_value_, bool has_past_key_value_, int layer_idx_)
#endif
        : hidden_states(hidden_states_),
          attention_mask(attention_mask_),
          past_key(past_key_),
          past_value(past_value_),
          has_past_key_value(has_past_key_value_),
          layer_idx(layer_idx_) {
    }
};

class Int4llamaAttention {
   public:
    Int4llamaAttention(std::string param_path, const struct model_config config, int layer_idx);
    Int4llamaAttention() {}
    static void initialized_memory(const struct model_config config);
    struct Int4llamaAttention_output forward(std::string param_path, const struct Int4llamaAttention_input &input);

#if !(DEC_SHARED_MEM)
    int *q_weight = nullptr, *k_weight = nullptr, *v_weight = nullptr, *o_weight = nullptr, *qkv_weight = nullptr;
#endif

#ifdef QM_CUDA
    void free_cuda_memory();
    half *cos_buf = nullptr, *sin_buf = nullptr;
#elif defined(QM_METAL)
    void metal_free();
    half *cos_buf = nullptr, *sin_buf = nullptr;
#else
    float *cos_buf = nullptr, *sin_buf = nullptr;
#endif

   private:
    std::string profile_name = "Int4llamaAttention";
    int embed_dim, num_heads, head_dim;
#ifdef QM_CUDA
    Linear_half_int4 o_proj, qkv_proj;
    RotaryPosEmb_cuda rotary_pos_emb;
    BMM_F16T qk_bmm, pv_bmm;
    int max_sqlen;
#elif defined(QM_METAL)
    Linear_half_int4 o_proj, qkv_proj;
    RotaryPosEmb_metal rotary_pos_emb;
    BMM_F16T qk_bmm, pv_bmm;
    int max_sqlen;
#else
    Linear_FP_int4 k_proj, v_proj, q_proj, o_proj, qkv_proj;
    RotaryPosEmb rotary_pos_emb;
    BMM_F32T qk_bmm, pv_bmm;
    void unshape(Matrix3D<float> shaped, Matrix3D<float> unshape, int sqlen);
    void shape(Matrix3D<float> unshape, Matrix3D<float> shaped, int sqlen);
    void shape_qkv(Matrix3D<float> unshape, Matrix3D<float> shaped_q, Matrix3D<float> shaped_k,
                                          Matrix3D<float> shaped_v, int sqlen);
#endif
};
