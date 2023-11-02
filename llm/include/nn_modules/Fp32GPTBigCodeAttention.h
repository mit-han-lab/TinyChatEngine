#include <utility>

#include "common.h"
#include "operators.h"

struct Fp32GPTBigCodeAttention_output {
    Matrix3D<float> attn_output;
    Matrix3D<float> attn_probs_reshaped;
    std::pair<Matrix3D<float>, Matrix3D<float>> past_key_value;
};
struct Fp32GPTBigCodeAttention_input {
    Matrix3D<float> hidden_states;
    Matrix3D<float> attention_mask;
    Matrix3D<float> past_key, past_value;
    bool has_past_key_value = false;
    int layer_idx;

    Fp32GPTBigCodeAttention_input(Matrix3D<float> hidden_states_, Matrix3D<float> attention_mask_, int layer_idx_)
        : hidden_states(hidden_states_), attention_mask(attention_mask_), layer_idx(layer_idx_) {}

    Fp32GPTBigCodeAttention_input(Matrix3D<float> hidden_states_, Matrix3D<float> attention_mask_, Matrix3D<float> past_key_,
                           Matrix3D<float> past_value_, bool has_past_key_value_, int layer_idx_)
        : hidden_states(hidden_states_),
          attention_mask(attention_mask_),
          past_key(past_key_),
          past_value(past_value_),
          has_past_key_value(has_past_key_value_),
          layer_idx(layer_idx_) {}
};

class Fp32GPTBigCodeAttention {
   public:
    Fp32GPTBigCodeAttention(std::string param_path, const struct model_config config);
    Fp32GPTBigCodeAttention() {}
    static void initialized_memory(const struct model_config config);
    struct Fp32GPTBigCodeAttention_output forward(const struct Fp32GPTBigCodeAttention_input &input);

   private:
    void unshape(Matrix3D<float> shaped, Matrix3D<float> unshape, int sqlen);
    void shape_qkv(Matrix3D<float> unshape, Matrix3D<float> shaped_q, Matrix3D<float> shaped_k,
                                          Matrix3D<float> shaped_v, int sqlen);
    float scaling;
    int embed_dim, num_heads, head_dim, kv_heads, kv_dim;
    BMM_F32T qk_bmm, pv_bmm;
    Linear_FP c_attn, c_proj;
    std::string profile_name = "Fp32GPTBigCodeAttention";
};
