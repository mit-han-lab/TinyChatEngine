#include "Fp32OPTDecoderLayer.h"

#include "utils.h"

// Shared memory space across all layers
static float *hidden_states_float_arr;
static float *final_layer_norm_arr;
static float *fc_1_arr;
static float *fc_2_arr;
static float *temp;
static float *hidden_states_arr;

template <typename T>
void add(Matrix3D<T> a, Matrix3D<T> b, Matrix3D<T> c) {
    PROFILE_START("Fp32OPTDecoderLayer::add");
    assert(c.length() == a.length() && a.length() == b.length());

    for (int i = 0; i < a.length(); i++) {
        c.m_data[i] = a.m_data[i] + b.m_data[i];
    }
    PROFILE_END("Fp32OPTDecoderLayer::add");
}

struct Fp32OPTDecoderLayer_output Fp32OPTDecoderLayer::forward(const struct Fp32OPTDecoderLayer_input &input) {
    PROFILE_START(profile_name);
    // Layernorm
    Matrix3D<float> hidden_states(hidden_states_arr, input.hidden_states.m_dim_x, input.hidden_states.m_dim_y,
                                  input.hidden_states.m_dim_z);
    this->self_attn_layer_norm.forward(input.hidden_states, hidden_states);

    // Attention
    struct Fp32OPTAttention_input attn_param(hidden_states, input.attention_mask, input.past_key, input.past_value,
                                             input.has_past_key_value, this->layer_idx);
    struct Fp32OPTAttention_output attn_output = this->attn.forward(attn_param);

    // Residual add
    Matrix3D<float> residual_add(hidden_states_float_arr, input.hidden_states.m_dim_x, input.hidden_states.m_dim_y,
                                 input.hidden_states.m_dim_z);
    add(input.hidden_states, attn_output.attn_output, residual_add);

    // Layernorm
    Matrix3D<float> final_layer_norm(final_layer_norm_arr, input.hidden_states.m_dim_x, input.hidden_states.m_dim_y,
                                     input.hidden_states.m_dim_z);
    this->final_layer_norm.forward(residual_add, final_layer_norm);

    // FC
    Matrix3D<float> fc1_out(fc_1_arr, input.hidden_states.m_dim_x, input.hidden_states.m_dim_y, this->hidden_dim);
    this->fc1.forward(final_layer_norm, fc1_out);
    // RELU
    for (int i = 0; i < fc1_out.length(); i++) {
        float v = fc1_out.m_data[i];
        if (v < 0) fc1_out.m_data[i] = 0;
    }

    Matrix3D<float> fc2_out(fc_2_arr, input.hidden_states.m_dim_x, input.hidden_states.m_dim_y,
                            input.hidden_states.m_dim_z);
    this->fc2.forward(fc1_out, fc2_out);

    // Reidual add
    add(residual_add, fc2_out, residual_add);

    struct Fp32OPTDecoderLayer_output output(residual_add, attn_output.attn_probs_reshaped, attn_output.past_key_value);
    PROFILE_END(profile_name);
    return output;
}

Fp32OPTDecoderLayer::Fp32OPTDecoderLayer(std::string param_path, const model_config config, int layer_idx) {
    if (layer_idx == 0) {
        allocate_aligned_memory(hidden_states_float_arr, config.max_sqlen * config.embed_dim * sizeof(float));
        allocate_aligned_memory(final_layer_norm_arr, config.max_sqlen * config.embed_dim * sizeof(float));
        allocate_aligned_memory(fc_1_arr, config.max_sqlen * config.hidden_dim * sizeof(float));
        allocate_aligned_memory(fc_2_arr, config.max_sqlen * config.embed_dim * sizeof(float));
        allocate_aligned_memory(hidden_states_arr, config.max_sqlen * config.embed_dim * sizeof(float));
        Fp32OPTAttention::initialized_memory(config);
    }

    struct LayerNorm_params self_attn_layer_norm, final_layer_norm;
    Matrix3D<float> self_attn_layer_norm_weight(new float[config.embed_dim], 1, 1, config.embed_dim);
    Matrix3D<float> self_attn_layer_norm_bias(new float[config.embed_dim], 1, 1, config.embed_dim);
    self_attn_layer_norm.weight = self_attn_layer_norm_weight;
    self_attn_layer_norm.bias = self_attn_layer_norm_bias;

    Matrix3D<float> final_layer_norm_weight(new float[config.embed_dim], 1, 1, config.embed_dim);
    Matrix3D<float> final_layer_norm_bias(new float[config.embed_dim], 1, 1, config.embed_dim);
    final_layer_norm.weight = final_layer_norm_weight;
    final_layer_norm.bias = final_layer_norm_bias;
    this->self_attn_layer_norm = LayerNorm(self_attn_layer_norm);
    load_LayerNorm(this->self_attn_layer_norm, param_path + "/self_attn_layer_norm");
    this->final_layer_norm = LayerNorm(final_layer_norm);
    load_LayerNorm(this->final_layer_norm, param_path + "/final_layer_norm");

    float *fc1_weight, *fc2_weight;
    allocate_aligned_memory(fc1_weight, (config.embed_dim * config.hidden_dim * sizeof(float)));
    allocate_aligned_memory(fc2_weight, (config.embed_dim * config.hidden_dim * sizeof(float)));
    float *fc1_bias, *fc2_bias;
    allocate_aligned_memory(fc1_bias, (config.hidden_dim * sizeof(float)));
    allocate_aligned_memory(fc2_bias, (config.embed_dim * sizeof(float)));
    this->fc1 =
        Linear_FP(Matrix3D<float>(fc1_weight, 1, config.hidden_dim, config.embed_dim), param_path + "/fc1/weight.bin",
                  Matrix3D<float>(fc1_bias, 1, 1, config.hidden_dim), param_path + "/fc1/bias.bin");
    this->fc2 =
        Linear_FP(Matrix3D<float>(fc2_weight, 1, config.embed_dim, config.hidden_dim), param_path + "/fc2/weight.bin",
                  Matrix3D<float>(fc2_bias, 1, 1, config.embed_dim), param_path + "/fc2/bias.bin");

    this->embed_dim = config.embed_dim;
    this->num_attention_heads = config.num_heads;
    this->hidden_dim = config.hidden_dim;
    this->layer_idx = layer_idx;

    this->attn = Fp32OPTAttention(param_path + "/self_attn", config);
}
