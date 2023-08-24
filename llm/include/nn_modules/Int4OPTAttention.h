#include <utility>

#include "common.h"
#include "operators.h"

struct Int4OPTAttention_output {
    Matrix3D<float> attn_output;
    Matrix3D<float> attn_probs_reshaped;
    std::pair<Matrix3D<float>, Matrix3D<float>> past_key_value;
};
struct Int4OPTAttention_input {
    Matrix3D<float> hidden_states;
    Matrix3D<float> attention_mask;
    Matrix3D<float> past_key, past_value;
    bool has_past_key_value = false;
    int layer_idx;

    Int4OPTAttention_input(Matrix3D<float> hidden_states_, Matrix3D<float> attention_mask_, int layer_idx_)
        : hidden_states(hidden_states_), attention_mask(attention_mask_), layer_idx(layer_idx_) {}

    Int4OPTAttention_input(Matrix3D<float> hidden_states_, Matrix3D<float> attention_mask_, Matrix3D<float> past_key_,
                           Matrix3D<float> past_value_, bool has_past_key_value_, int layer_idx_)
        : hidden_states(hidden_states_),
          attention_mask(attention_mask_),
          past_key(past_key_),
          past_value(past_value_),
          has_past_key_value(has_past_key_value_),
          layer_idx(layer_idx_) {}
};

class Int4OPTAttention {
   public:
    Int4OPTAttention(std::string param_path, const struct model_config config);
    Int4OPTAttention() {}
    static void initialized_memory(const struct model_config config);
    struct Int4OPTAttention_output forward(const struct Int4OPTAttention_input &input);

   private:
    void unshape(Matrix3D<float> shaped, Matrix3D<float> unshape, int sqlen);
    void shpae(Matrix3D<float> unshape, Matrix3D<float> shaped, int sqlen);
    float scaling;
    int embed_dim, num_heads, head_dim;
    BMM_F32T qk_bmm, pv_bmm;
    Linear_FP_int4 k_proj, v_proj, q_proj, out_proj;
    std::string profile_name = "Int4OPTAttention";
};
