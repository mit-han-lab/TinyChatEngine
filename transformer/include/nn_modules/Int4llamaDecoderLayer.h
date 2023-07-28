#include "Int4llamaAttention.h"
#include "common.h"
#include "operators.h"

struct Int4llamaDecoderLayer_output {
    Matrix3D<float> hidden_states;
    Matrix3D<float> attentions;
    std::pair<Matrix3D<float>, Matrix3D<float>> past_key_value;

    Int4llamaDecoderLayer_output(Matrix3D<float> hidden_states_, Matrix3D<float> attentions_,
                                 std::pair<Matrix3D<float>, Matrix3D<float>> past_key_value_) {
        hidden_states = hidden_states_;
        attentions = attentions_;
        past_key_value = past_key_value_;
    };
};
struct Int4llamaDecoderLayer_input {
    Matrix3D<float> hidden_states;
    Matrix3D<float> attention_mask;
    Matrix3D<float> past_key, past_value;
    bool has_past_key_value = false;

    Int4llamaDecoderLayer_input(Matrix3D<float> &hidden_states_, Matrix3D<float> &attention_mask_) {
        hidden_states = hidden_states_;
        attention_mask = attention_mask_;
        has_past_key_value = false;
    }

    Int4llamaDecoderLayer_input(Matrix3D<float> &hidden_states_, Matrix3D<float> &attention_mask_,
                                Matrix3D<float> past_key_, Matrix3D<float> past_value_) {
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
    struct Int4llamaDecoderLayer_output forward(const struct Int4llamaDecoderLayer_input &input);

    int embed_dim, num_attention_heads, hidden_dim, layer_idx;
    LlamaRMSNorm input_layernorm, post_attention_layernorm;  // from torch_int.nn
    Linear_FP_int4 gate_proj, down_proj, up_proj;
    Int4llamaAttention attn;
    std::string profile_name = "Int4llamaDecoderLayer";
};
