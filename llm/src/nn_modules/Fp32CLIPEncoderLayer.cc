#include "Fp32CLIPEncoderLayer.h"
#include "utils.h"

template <typename T>
void add(Matrix3D<T> a, Matrix3D<T> b, Matrix3D<T> c) {
    PROFILE_START("Fp32CLIPEncoderLayer::add");
    assert(c.length() == a.length() && a.length() == b.length());

    for (int i = 0; i < a.length(); i++) {
        c.m_data[i] = a.m_data[i] + b.m_data[i];
    }
    PROFILE_END("Fp32CLIPEncoderLayer::add");
}

// Shared memory space across all layers
static float *hidden_states_float_arr;
static float *final_layer_norm_arr;
static float *mlp_fc1_arr;
static float *mlp_fc2_arr;
static float *temp;
static float *hidden_states_arr;

Fp32CLIPEncoderLayer::Fp32CLIPEncoderLayer(std::string param_path, const struct model_config config, int layer_idx) {
    if (layer_idx == 0) {
        allocate_aligned_memory(hidden_states_float_arr, config.max_sqlen * config.embed_dim * sizeof(float));
        allocate_aligned_memory(final_layer_norm_arr, config.max_sqlen * config.embed_dim * sizeof(float));
        allocate_aligned_memory(mlp_fc1_arr, config.max_sqlen * config.hidden_dim * sizeof(float));
        allocate_aligned_memory(mlp_fc2_arr, config.max_sqlen * config.embed_dim * sizeof(float));
        allocate_aligned_memory(hidden_states_arr, config.max_sqlen * config.embed_dim * sizeof(float));
        Fp32CLIPAttention::initialized_memory(config);
    }

    struct LayerNorm_params layer_norm1, layer_norm2;
    float *layer_norm1_weight_buf, *layer_norm1_bias_buf;
    allocate_aligned_memory(layer_norm1_weight_buf, config.embed_dim * sizeof(float));
    allocate_aligned_memory(layer_norm1_bias_buf, config.embed_dim * sizeof(float));
    Matrix3D<float> layer_norm1_weight(layer_norm1_weight_buf, 1, 1, config.embed_dim);
    Matrix3D<float> layer_norm1_bias(layer_norm1_bias_buf, 1, 1, config.embed_dim);
    layer_norm1.weight = layer_norm1_weight;
    layer_norm1.bias = layer_norm1_bias;
    float *layer_norm2_weight_buf, *layer_norm2_bias_buf;
    allocate_aligned_memory(layer_norm2_weight_buf, config.embed_dim * sizeof(float));
    allocate_aligned_memory(layer_norm2_bias_buf, config.embed_dim * sizeof(float));
    Matrix3D<float> layer_norm2_weight(layer_norm2_weight_buf, 1, 1, config.embed_dim);
    Matrix3D<float> layer_norm2_bias(layer_norm2_bias_buf, 1, 1, config.embed_dim);
    layer_norm2.weight = layer_norm2_weight;
    layer_norm2.bias = layer_norm2_bias;

    this->layer_norm1 = LayerNorm(layer_norm1);
    load_LayerNorm(this->layer_norm1, param_path + "/layer_norm1");
    this->layer_norm2 = LayerNorm(layer_norm2);
    load_LayerNorm(this->layer_norm2, param_path + "/layer_norm2");

    this->embed_dim = config.embed_dim;
    this->num_attention_heads = config.num_heads;
    this->hidden_dim = config.hidden_dim;
    this->layer_idx = layer_idx;

    this->attn = Fp32CLIPAttention(param_path + "/self_attn", config);

    float *mlp_fc1_weight, *mlp_fc2_weight;
    allocate_aligned_memory(mlp_fc1_weight, config.embed_dim * config.hidden_dim * sizeof(float));
    allocate_aligned_memory(mlp_fc2_weight, config.hidden_dim * config.embed_dim * sizeof(float));
    float *mlp_fc1_bias, *mlp_fc2_bias;
    allocate_aligned_memory(mlp_fc1_bias, (config.hidden_dim * sizeof(float)));
    allocate_aligned_memory(mlp_fc2_bias, (config.embed_dim * sizeof(float)));
    this->mlp_fc1 = 
        Linear_FP(Matrix3D<float>(mlp_fc1_weight, 1, config.hidden_dim, config.embed_dim), param_path + "/mlp_fc1/weight.bin",
                  Matrix3D<float>(mlp_fc1_bias, 1, 1, config.hidden_dim), (param_path + "/mlp_fc1/bias.bin"));
    this->mlp_fc1.has_bias = true;
    this->mlp_fc2 = 
        Linear_FP(Matrix3D<float>(mlp_fc2_weight, 1, config.embed_dim, config.hidden_dim), param_path + "/mlp_fc2/weight.bin",
                  Matrix3D<float>(mlp_fc2_bias, 1, 1, config.embed_dim), (param_path + "/mlp_fc2/bias.bin"));
    this->mlp_fc2.has_bias = true;
}

struct Fp32CLIPEncoderLayer_output Fp32CLIPEncoderLayer::forward(const struct Fp32CLIPEncoderLayer_input &input) {
    PROFILE_START(profile_name);

    // Layernorm 1
    Matrix3D<float> hidden_states(hidden_states_arr, input.hidden_states.m_dim_x, input.hidden_states.m_dim_y,
                                  input.hidden_states.m_dim_z);
    this->layer_norm1.forward(input.hidden_states, hidden_states);

    // Attention
    struct Fp32CLIPAttention_input attn_param(hidden_states, input.attention_mask, input.past_key, input.past_value,
                                               input.has_past_key_value, this->layer_idx);
    struct Fp32CLIPAttention_output attn_output = this->attn.forward(attn_param);

    // Residual add
    Matrix3D<float> residual_add(hidden_states_float_arr, input.hidden_states.m_dim_x, input.hidden_states.m_dim_y,
                                 input.hidden_states.m_dim_z);
    add(input.hidden_states, attn_output.attn_output, residual_add);

    // Layernorm 2
    Matrix3D<float> post_attention_layernorm(final_layer_norm_arr, input.hidden_states.m_dim_x,
                                             input.hidden_states.m_dim_y, input.hidden_states.m_dim_z);
    this->layer_norm2.forward(residual_add, post_attention_layernorm);

    // mlp_fc1: embed_dim -> hidden_dim
    Matrix3D<float> mlp_fc1(mlp_fc1_arr, input.hidden_states.m_dim_x, input.hidden_states.m_dim_y,
                              this->hidden_dim);
    this->mlp_fc1.forward(post_attention_layernorm, mlp_fc1);

    // Quick GELU
    Gelu_quick(mlp_fc1);

    // mlp_fc2: hidden_dim -> embed_dim
    Matrix3D<float> mlp_fc2(mlp_fc2_arr, input.hidden_states.m_dim_x, input.hidden_states.m_dim_y, this->embed_dim);
    this->mlp_fc2.forward(mlp_fc1, mlp_fc2);

    // Residual add
    add(residual_add, mlp_fc2, residual_add);

    struct Fp32CLIPEncoderLayer_output output(residual_add, attn_output.attn_probs_reshaped,
                                               attn_output.past_key_value);
    PROFILE_END(profile_name);
    return output;
}
