#include "Int4GPTBigCodeAttention.h"
#include "common.h"
#include "operators.h"

struct Int4GPTBigCodeDecoderLayer_output {
    Matrix3D<float> hidden_states;
    Matrix3D<float> attentions;
    std::pair<Matrix3D<float>, Matrix3D<float>> past_key_value;

    Int4GPTBigCodeDecoderLayer_output(Matrix3D<float> hidden_states_, Matrix3D<float> attentions_,
                               std::pair<Matrix3D<float>, Matrix3D<float>> past_key_value_) {
        hidden_states = hidden_states_;
        attentions = attentions_;
        past_key_value = past_key_value_;
    };
};
struct Int4GPTBigCodeDecoderLayer_input {
    Matrix3D<float> hidden_states;
    Matrix3D<float> attention_mask;
    Matrix3D<float> past_key, past_value;
    bool has_past_key_value = false;

    Int4GPTBigCodeDecoderLayer_input(Matrix3D<float> &hidden_states_, Matrix3D<float> &attention_mask_) {
        hidden_states = hidden_states_;
        attention_mask = attention_mask_;
        has_past_key_value = false;
    }

    Int4GPTBigCodeDecoderLayer_input(Matrix3D<float> &hidden_states_, Matrix3D<float> &attention_mask_,
                              Matrix3D<float> past_key_, Matrix3D<float> past_value_) {
        hidden_states = hidden_states_;
        attention_mask = attention_mask_;
        past_key = past_key_;
        past_value = past_value_;
        has_past_key_value = true;
    }
};

class Int4GPTBigCodeDecoderLayer {
   public:
    Int4GPTBigCodeDecoderLayer(std::string param_path, const struct model_config config, int layer_idx);
    struct Int4GPTBigCodeDecoderLayer_output forward(const struct Int4GPTBigCodeDecoderLayer_input &input);

    int embed_dim, num_attention_heads, hidden_dim, layer_idx;
    LayerNorm ln_1, ln_2;  // from torch_int.nn
    Linear_FP_int4 fc1, fc2;
    Int4GPTBigCodeAttention attn;
    std::string profile_name = "Int4GPTBigCodeDecoderLayer";
};
