#include "Fp32CLIPAttention.h"
#include "common.h"
#include "operators.h"

struct Fp32CLIPEncoderLayer_output {
    Matrix3D<float> hidden_states;
    Matrix3D<float> attentions;
    std::pair<Matrix3D<float>, Matrix3D<float>> past_key_value;

    Fp32CLIPEncoderLayer_output(Matrix3D<float> hidden_states_, Matrix3D<float> attentions_,
                                 std::pair<Matrix3D<float>, Matrix3D<float>> past_key_value_) {
        hidden_states = hidden_states_;
        attentions = attentions_;
        past_key_value = past_key_value_;
    };
};
struct Fp32CLIPEncoderLayer_input {
    Matrix3D<float> hidden_states;
    Matrix3D<float> attention_mask;
    Matrix3D<float> past_key, past_value;
    bool has_past_key_value = false;

    Fp32CLIPEncoderLayer_input(Matrix3D<float> &hidden_states_, Matrix3D<float> attention_mask_) {
        hidden_states = hidden_states_;
        attention_mask = attention_mask_;
        has_past_key_value = false;
    }

    Fp32CLIPEncoderLayer_input(Matrix3D<float> &hidden_states_, Matrix3D<float> attention_mask_,
                                Matrix3D<float> past_key_, Matrix3D<float> past_value_) {
        hidden_states = hidden_states_;
        attention_mask = attention_mask_;
        past_key = past_key_;
        past_value = past_value_;
        has_past_key_value = true;
    }
};

class Fp32CLIPEncoderLayer {
   public:
    Fp32CLIPEncoderLayer(std::string param_path, const struct model_config config, int layer_idx);
    struct Fp32CLIPEncoderLayer_output forward(const struct Fp32CLIPEncoderLayer_input &input);

    int embed_dim, num_attention_heads, hidden_dim, layer_idx;
    LayerNorm layer_norm1, layer_norm2;
    Linear_FP mlp_fc1, mlp_fc2;
    Fp32CLIPAttention attn;
    std::string profile_name = "Fp32CLIPEncoderLayer";
};
