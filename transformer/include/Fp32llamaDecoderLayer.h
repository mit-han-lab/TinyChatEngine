#include "Fp32llamaAttention.h"
#include "common.h"
#include "operators.h"

struct Fp32llamaDecoderLayer_output {
    Matrix3D<float> hidden_states;
    Matrix3D<float> attentions;
    std::pair<Matrix3D<float>, Matrix3D<float>> past_key_value;

    Fp32llamaDecoderLayer_output(Matrix3D<float> hidden_states_, Matrix3D<float> attentions_,
                                 std::pair<Matrix3D<float>, Matrix3D<float>> past_key_value_) {
        hidden_states = hidden_states_;
        attentions = attentions_;
        past_key_value = past_key_value_;
    };
};
struct Fp32llamaDecoderLayer_input {
    Matrix3D<float> hidden_states;
    Matrix3D<float> attention_mask;
    Matrix3D<float> past_key, past_value;
    bool has_past_key_value = false;

    Fp32llamaDecoderLayer_input(Matrix3D<float> &hidden_states_, Matrix3D<float> &attention_mask_) {
        hidden_states = hidden_states_;
        attention_mask = attention_mask_;
        has_past_key_value = false;
    }

    Fp32llamaDecoderLayer_input(Matrix3D<float> &hidden_states_, Matrix3D<float> &attention_mask_,
                                Matrix3D<float> past_key_, Matrix3D<float> past_value_) {
        hidden_states = hidden_states_;
        attention_mask = attention_mask_;
        past_key = past_key_;
        past_value = past_value_;
        has_past_key_value = true;
    }
};

class Fp32llamaDecoderLayer {
   public:
    Fp32llamaDecoderLayer(std::string param_path, const struct model_config config, int layer_idx);
    struct Fp32llamaDecoderLayer_output forward(const struct Fp32llamaDecoderLayer_input &input);

    int embed_dim, num_attention_heads, hidden_dim, layer_idx;
    LlamaRMSNorm input_layernorm, post_attention_layernorm;
    Linear_FP gate_proj, down_proj, up_proj;
    Fp32llamaAttention attn;
    std::string profile_name = "Fp32llamaDecoderLayer";
};
