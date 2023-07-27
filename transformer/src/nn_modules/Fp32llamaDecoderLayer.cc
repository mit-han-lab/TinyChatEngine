#include "Fp32llamaDecoderLayer.h"

#include "utils.h"

template <typename T>
void add(Matrix3D<T> a, Matrix3D<T> b, Matrix3D<T> c) {
    PROFILE_START("int4llamaDecoderLayer::add");
    assert(c.length() == a.length() && a.length() == b.length());

    for (int i = 0; i < a.length(); i++) {
        c.m_data[i] = a.m_data[i] + b.m_data[i];
    }
    PROFILE_END("int4llamaDecoderLayer::add");
}

void SiLuMul(Matrix3D<float> a, Matrix3D<float> b) {
    PROFILE_START("MulSiLu");
    for (int i = 0; i < a.length(); i++) {
        float v = a.m_data[i];
        float silu_v = v * (1.0 / (1.0 + exp(-1 * v)));
        a.m_data[i] = silu_v * b.m_data[i];
    }
    PROFILE_END("MulSiLu");
}

// Shared memory space across all layers
static float *hidden_states_float_arr;
static float *final_layer_norm_arr;
static float *gate_proj_arr;
static float *up_proj_arr;
static float *down_proj_arr;
static float *temp;
static float *hidden_states_arr;

struct Fp32llamaDecoderLayer_output Fp32llamaDecoderLayer::forward(const struct Fp32llamaDecoderLayer_input &input) {
    PROFILE_START(profile_name);
    // Layernorm
    Matrix3D<float> hidden_states(hidden_states_arr, input.hidden_states.m_dim_x, input.hidden_states.m_dim_y,
                                  input.hidden_states.m_dim_z);
    this->input_layernorm.forward(input.hidden_states, hidden_states);

    // Attention
    struct Fp32llamaAttention_input attn_param(hidden_states, input.attention_mask, input.past_key, input.past_value,
                                               input.has_past_key_value, this->layer_idx);
    struct Fp32llamaAttention_output attn_output = this->attn.forward(attn_param);

    // Residual add
    Matrix3D<float> residual_add(hidden_states_float_arr, input.hidden_states.m_dim_x, input.hidden_states.m_dim_y,
                                 input.hidden_states.m_dim_z);
    add(input.hidden_states, attn_output.attn_output, residual_add);

    // Layernorm
    Matrix3D<float> post_attention_layernorm(final_layer_norm_arr, input.hidden_states.m_dim_x,
                                             input.hidden_states.m_dim_y, input.hidden_states.m_dim_z);
    this->post_attention_layernorm.forward(residual_add, post_attention_layernorm);

    // Gate proj: embedding -> hidden_dim
    Matrix3D<float> gate_proj(gate_proj_arr, input.hidden_states.m_dim_x, input.hidden_states.m_dim_y,
                              this->hidden_dim);
    this->gate_proj.forward(post_attention_layernorm, gate_proj);

    // up proj: embedding -> hidden_dim
    Matrix3D<float> up_proj(up_proj_arr, input.hidden_states.m_dim_x, input.hidden_states.m_dim_y, this->hidden_dim);
    this->up_proj.forward(post_attention_layernorm, up_proj);

    // silu
    SiLuMul(gate_proj, up_proj);

    // down proj: hidden_dim -> embedding
    Matrix3D<float> down_proj(down_proj_arr, input.hidden_states.m_dim_x, input.hidden_states.m_dim_y, this->embed_dim);
    this->down_proj.forward(gate_proj, down_proj);

    // Residual add
    add(residual_add, down_proj, residual_add);

    struct Fp32llamaDecoderLayer_output output(residual_add, attn_output.attn_probs_reshaped,
                                               attn_output.past_key_value);
    PROFILE_END(profile_name);
    return output;
}

Fp32llamaDecoderLayer::Fp32llamaDecoderLayer(std::string param_path, const struct model_config config, int layer_idx) {
    if (layer_idx == 0) {
        allocate_aligned_memory(hidden_states_float_arr, config.max_sqlen * config.embed_dim * sizeof(float));
        allocate_aligned_memory(final_layer_norm_arr, config.max_sqlen * config.embed_dim * sizeof(float));
        allocate_aligned_memory(gate_proj_arr, config.max_sqlen * config.hidden_dim * sizeof(float));
        allocate_aligned_memory(up_proj_arr, config.max_sqlen * config.hidden_dim * sizeof(float));
        allocate_aligned_memory(down_proj_arr, config.max_sqlen * config.embed_dim * sizeof(float));
        allocate_aligned_memory(hidden_states_arr, config.max_sqlen * config.embed_dim * sizeof(float));
        Fp32llamaAttention::initialized_memory(config);
    }

    float *input_layernorm_weight_ptr;
    allocate_aligned_memory(input_layernorm_weight_ptr, config.embed_dim * sizeof(float));
    Matrix3D<float> input_layernorm_weight(input_layernorm_weight_ptr, 1, 1, config.embed_dim);
    input_layernorm_weight.load((param_path + "/input_layernorm/weight.bin").c_str());
    this->input_layernorm = LlamaRMSNorm(input_layernorm_weight);

    float *post_attention_layernorm_ptr;
    allocate_aligned_memory(post_attention_layernorm_ptr, config.embed_dim * sizeof(float));
    Matrix3D<float> post_attention_layernorm_weight(post_attention_layernorm_ptr, 1, 1, config.embed_dim);
    post_attention_layernorm_weight.load((param_path + "/post_attention_layernorm/weight.bin").c_str());
    this->post_attention_layernorm = LlamaRMSNorm(post_attention_layernorm_weight);

    this->embed_dim = config.embed_dim;
    this->num_attention_heads = config.num_heads;
    this->hidden_dim = config.hidden_dim;
    this->layer_idx = layer_idx;

    this->attn = Fp32llamaAttention(param_path + "/self_attn", config);

    float *gate_proj_weight, *down_proj_weight, *up_proj_weight;
    allocate_aligned_memory(gate_proj_weight, config.embed_dim * config.hidden_dim * sizeof(float));
    allocate_aligned_memory(down_proj_weight, config.hidden_dim * config.embed_dim * sizeof(float));
    allocate_aligned_memory(up_proj_weight, config.embed_dim * config.hidden_dim * sizeof(float));
    this->gate_proj = Linear_FP(Matrix3D<float>(gate_proj_weight, 1, config.hidden_dim, config.embed_dim),
                                (param_path + "/gate_proj/weight.bin"));
    this->down_proj = Linear_FP(Matrix3D<float>(down_proj_weight, 1, config.embed_dim, config.hidden_dim),
                                (param_path + "/down_proj/weight.bin"));
    this->up_proj = Linear_FP(Matrix3D<float>(up_proj_weight, 1, config.hidden_dim, config.embed_dim),
                              (param_path + "/up_proj/weight.bin"));
}
