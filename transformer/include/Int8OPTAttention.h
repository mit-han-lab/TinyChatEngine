#include <utility>

#include "common.h"
#include "operators.h"

struct Int8OPTAttention_output {
    Matrix3D<float> attn_output;
    Matrix3D<int8_t> attn_probs_reshaped;
    std::pair<Matrix3D<int8_t>, Matrix3D<int8_t>> past_key_value;
};
struct Int8OPTAttention_input {
    Matrix3D<int8_t> hidden_states;
    Matrix3D<float> attention_mask;
    Matrix3D<int8_t> past_key, past_value;
    bool has_past_key_value = false;
    int layer_idx;

    Int8OPTAttention_input(Matrix3D<int8_t> hidden_states_, Matrix3D<float> attention_mask_, int layer_idx_)
        : hidden_states(hidden_states_), attention_mask(attention_mask_), layer_idx(layer_idx_) {}

    Int8OPTAttention_input(Matrix3D<int8_t> hidden_states_, Matrix3D<float> attention_mask_, Matrix3D<int8_t> past_key_,
                           Matrix3D<int8_t> past_value_, bool has_past_key_value_, int layer_idx_)
        : hidden_states(hidden_states_),
          attention_mask(attention_mask_),
          past_key(past_key_),
          past_value(past_value_),
          has_past_key_value(has_past_key_value_),
          layer_idx(layer_idx_) {}
};

class Int8OPTAttention {
   public:
    Int8OPTAttention(std::string param_path, const struct model_config config, BMM_S8T_S8N_F32T &qk_bmm,
                     BMM_S8T_S8N_S8T &pv_bmm, W8A8B8O8Linear &k_proj, W8A8B8O8Linear &v_proj, W8A8B8O8Linear &q_proj,
                     W8A8BFP32OFP32Linear &out_proj);
    Int8OPTAttention() {}
    static void initialized_memory(const struct model_config config);
    struct Int8OPTAttention_output forward(const struct Int8OPTAttention_input &input);

   private:
    void unshape(Matrix3D<int8_t> shaped, Matrix3D<int8_t> unshape, int sqlen);
    void shpae(Matrix3D<int8_t> unshape, Matrix3D<int8_t> shaped, int sqlen);
    int embed_dim, num_heads, head_dim;
    BMM_S8T_S8N_F32T qk_bmm;
    BMM_S8T_S8N_S8T pv_bmm;
    W8A8B8O8Linear k_proj, v_proj, q_proj;
    W8A8BFP32OFP32Linear out_proj;
    std::string profile_name = "Int8OPTAttention";
};
