#include <utility>

#include "common.h"
#include "operators.h"
#include "linear.cuh"

#include "operators.cuh"

struct Int4llamaAttention_output {
    Matrix3D<float16_t> attn_output;
    Matrix3D<float16_t> attn_probs_reshaped;
    std::pair<Matrix3D<float16_t>, Matrix3D<float16_t>> past_key_value;
};
struct Int4llamaAttention_input {
    Matrix3D<float16_t> hidden_states;
    Matrix3D<float16_t> attention_mask;
    Matrix3D<float16_t> past_key, past_value;
    bool has_past_key_value = false;
    int layer_idx;

    Int4llamaAttention_input(Matrix3D<float16_t> hidden_states_, Matrix3D<float16_t> attention_mask_, int layer_idx_)
        : hidden_states(hidden_states_), attention_mask(attention_mask_), layer_idx(layer_idx_) {}

    Int4llamaAttention_input(Matrix3D<float16_t> hidden_states_, Matrix3D<float16_t> attention_mask_, Matrix3D<float16_t> past_key_,
                             Matrix3D<float16_t> past_value_, bool has_past_key_value_, int layer_idx_)
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
    struct Int4llamaAttention_output forward(const struct Int4llamaAttention_input &input);

   private:
    void unshape(Matrix3D<float> shaped, Matrix3D<float> unshape, int sqlen);
    void shape(Matrix3D<float> unshape, Matrix3D<float> shaped, int sqlen);
    int embed_dim, num_heads, head_dim, max_sqlen;
    Linear_half_int4 k_proj, v_proj, q_proj, o_proj;
    RotaryPosEmb_cuda rotary_pos_emb;
    BMM_F16T qk_bmm, pv_bmm;
    std::string profile_name = "Int4llamaAttention";

    int *q_weight, *k_weight, *v_weight, *o_weight;
};
