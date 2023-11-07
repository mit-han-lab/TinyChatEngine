#include "Int4GPTBigCodeDecoderLayer.h"

#include "utils.h"

// Shared memory space across all layers
static float *hidden_states_float_arr;
static float *ln_2_arr;
static float *fc_1_arr;
static float *fc_2_arr;
static float *temp;
static float *hidden_states_arr;

template <typename T>
void add(Matrix3D<T> a, Matrix3D<T> b, Matrix3D<T> c) {
    PROFILE_START("Int4GPTBigCodeDecoderLayer::add");
    assert(c.length() == a.length() && a.length() == b.length());

    for (int i = 0; i < a.length(); i++) {
        c.m_data[i] = a.m_data[i] + b.m_data[i];
    }
    PROFILE_END("Int4GPTBigCodeDecoderLayer::add");
}

static const float GELU_COEF_A    = 0.044715f;
static const float SQRT_2_OVER_PI = 0.79788456080286535587989211986876f;
inline static float Gelu_imp(float x) {
    return 0.5f * x * (1.0f + tanhf(SQRT_2_OVER_PI * x * (1.0f + GELU_COEF_A * x * x)));
}
inline static void Gelu(Matrix3D<float> a) {
    PROFILE_START("Int4GPTBigCodeDecoderLayer::Gelu");
    for (int i = 0; i < a.length(); i++) {
        a.m_data[i] = Gelu_imp(a.m_data[i]);
    }
    PROFILE_END("Int4GPTBigCodeDecoderLayer::Gelu");
}

Int4GPTBigCodeDecoderLayer::Int4GPTBigCodeDecoderLayer(std::string param_path, const model_config config, int layer_idx) {
    if (layer_idx == 0) {
        allocate_aligned_memory(hidden_states_float_arr, config.max_sqlen * config.embed_dim * sizeof(float));
        allocate_aligned_memory(ln_2_arr, config.max_sqlen * config.embed_dim * sizeof(float));
        allocate_aligned_memory(fc_1_arr, config.max_sqlen * config.hidden_dim * sizeof(float));
        allocate_aligned_memory(fc_2_arr, config.max_sqlen * config.embed_dim * sizeof(float));
        allocate_aligned_memory(hidden_states_arr, config.max_sqlen * config.embed_dim * sizeof(float));
        Int4GPTBigCodeAttention::initialized_memory(config);
    }

    struct LayerNorm_params ln_1, ln_2;
    Matrix3D<float> ln_1_weight(new float[config.embed_dim], 1, 1, config.embed_dim);
    Matrix3D<float> ln_1_bias(new float[config.embed_dim], 1, 1, config.embed_dim);
    ln_1.weight = ln_1_weight;
    ln_1.bias = ln_1_bias;

    Matrix3D<float> ln_2_weight(new float[config.embed_dim], 1, 1, config.embed_dim);
    Matrix3D<float> ln_2_bias(new float[config.embed_dim], 1, 1, config.embed_dim);
    ln_2.weight = ln_2_weight;
    ln_2.bias = ln_2_bias;
    this->ln_1 = LayerNorm(ln_1);
    load_LayerNorm(this->ln_1, param_path + "/ln_1");
    this->ln_2 = LayerNorm(ln_2);
    load_LayerNorm(this->ln_2, param_path + "/ln_2");

    uint8_t *fc1_weight, *fc2_weight;
    allocate_aligned_memory(fc1_weight, (config.embed_dim * config.hidden_dim * sizeof(uint8_t) / 2));
    allocate_aligned_memory(fc2_weight, (config.embed_dim * config.hidden_dim * sizeof(uint8_t) / 2));
    float *fc1_bias, *fc2_bias;
    allocate_aligned_memory(fc1_bias, (config.hidden_dim * sizeof(float)));
    allocate_aligned_memory(fc2_bias, (config.embed_dim * sizeof(float)));
    this->fc1 =
        Linear_FP_int4(Matrix3D<uint8_t>(fc1_weight, 1, config.hidden_dim, config.embed_dim / 2), param_path + "/c_fc",
                  Matrix3D<float>(fc1_bias, 1, 1, config.hidden_dim), param_path + "/c_fc/bias.bin");
    this->fc1.has_bias = true;
    this->fc2 =
        Linear_FP_int4(Matrix3D<uint8_t>(fc2_weight, 1, config.embed_dim, config.hidden_dim / 2), param_path + "/c_proj",
                  Matrix3D<float>(fc2_bias, 1, 1, config.embed_dim), param_path + "/c_proj/bias.bin");
    this->fc2.has_bias = true;

    this->embed_dim = config.embed_dim;
    this->num_attention_heads = config.num_heads;
    this->hidden_dim = config.hidden_dim;
    this->layer_idx = layer_idx;

    this->attn = Int4GPTBigCodeAttention(param_path + "/attn", config);
}

struct Int4GPTBigCodeDecoderLayer_output Int4GPTBigCodeDecoderLayer::forward(const struct Int4GPTBigCodeDecoderLayer_input &input) {
    PROFILE_START(profile_name);
    // Layernorm
    // printf(("Before ln_1\n");
    Matrix3D<float> hidden_states(hidden_states_arr, input.hidden_states.m_dim_x, input.hidden_states.m_dim_y,
                                  input.hidden_states.m_dim_z);
    this->ln_1.forward(input.hidden_states, hidden_states);

    // Attention
    // printf(("Before attn\n");
    struct Int4GPTBigCodeAttention_input attn_param(hidden_states, input.attention_mask, input.past_key, input.past_value,
                                             input.has_past_key_value, this->layer_idx);
    struct Int4GPTBigCodeAttention_output attn_output = this->attn.forward(attn_param);
    // printf(("After attn\n");

    // Residual add
    Matrix3D<float> residual_add(hidden_states_float_arr, input.hidden_states.m_dim_x, input.hidden_states.m_dim_y,
                                 input.hidden_states.m_dim_z);
    add(input.hidden_states, attn_output.attn_output, residual_add);

    // Layernorm
    Matrix3D<float> ln_2(ln_2_arr, input.hidden_states.m_dim_x, input.hidden_states.m_dim_y,
                                     input.hidden_states.m_dim_z);
    this->ln_2.forward(residual_add, ln_2);

    // FC 1
    Matrix3D<float> fc1_out(fc_1_arr, input.hidden_states.m_dim_x, input.hidden_states.m_dim_y, this->hidden_dim);
    this->fc1.forward(ln_2, fc1_out);

    // GELU
    Gelu(fc1_out);

    // FC 2
    Matrix3D<float> fc2_out(fc_2_arr, input.hidden_states.m_dim_x, input.hidden_states.m_dim_y,
                            input.hidden_states.m_dim_z);
    this->fc2.forward(fc1_out, fc2_out);

    // Reidual add
    add(residual_add, fc2_out, residual_add);

    struct Int4GPTBigCodeDecoderLayer_output output(residual_add, attn_output.attn_probs_reshaped, attn_output.past_key_value);
    PROFILE_END(profile_name);
    return output;
}
