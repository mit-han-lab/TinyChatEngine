#include "Int8OPTAttention.h"
#include "common.h"
#include "operators.h"

struct Int8OPTDecoderLayer_output {
    Matrix3D<float> hidden_states;
    Matrix3D<int8_t> attentions;
    std::pair<Matrix3D<int8_t>, Matrix3D<int8_t>> past_key_value;

    Int8OPTDecoderLayer_output(Matrix3D<float> hidden_states_, Matrix3D<int8_t> attentions_,
                               std::pair<Matrix3D<int8_t>, Matrix3D<int8_t>> past_key_value_) {
        hidden_states = hidden_states_;
        attentions = attentions_;
        past_key_value = past_key_value_;
    };
};
struct Int8OPTDecoderLayer_input {
    Matrix3D<float> hidden_states;
    Matrix3D<float> attention_mask;
    Matrix3D<int8_t> past_key, past_value;
    bool has_past_key_value = false;

    Int8OPTDecoderLayer_input(Matrix3D<float> &hidden_states_, Matrix3D<float> &attention_mask_) {
        hidden_states = hidden_states_;
        attention_mask = attention_mask_;
        has_past_key_value = false;
    }

    Int8OPTDecoderLayer_input(Matrix3D<float> &hidden_states_, Matrix3D<float> &attention_mask_,
                              Matrix3D<int8_t> past_key_, Matrix3D<int8_t> past_value_) {
        hidden_states = hidden_states_;
        attention_mask = attention_mask_;
        past_key = past_key_;
        past_value = past_value_;
        has_past_key_value = true;
    }
};

class Int8OPTDecoderLayer {
   public:
    Int8OPTDecoderLayer(std::string param_path, const struct model_config config, int layer_idx,
                        LayerNormQ self_attn_layer_norm, LayerNormQ final_layer_norm, W8A8B8O8LinearReLU fc1,
                        W8A8BFP32OFP32Linear fc2, BMM_S8T_S8N_F32T qk_bmm, BMM_S8T_S8N_S8T pv_bmm,
                        W8A8B8O8Linear k_proj, W8A8B8O8Linear v_proj, W8A8B8O8Linear q_proj,
                        W8A8BFP32OFP32Linear out_proj);
    struct Int8OPTDecoderLayer_output forward(const struct Int8OPTDecoderLayer_input &input);

    int embed_dim, num_attention_heads, hidden_dim, layer_idx;
    LayerNormQ self_attn_layer_norm, final_layer_norm;  // from torch_int.nn
    W8A8B8O8LinearReLU fc1;
    W8A8BFP32OFP32Linear fc2;
    Int8OPTAttention attn;
    std::string profile_name = "Int8OPTDecoderLayer";
};
