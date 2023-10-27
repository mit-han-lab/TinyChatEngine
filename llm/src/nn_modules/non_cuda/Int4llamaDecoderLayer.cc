#include "Int4llamaDecoderLayer.h"

#include "utils.h"

// Shared memory space across all layers
static float *hidden_states_float_arr;
static float *final_layer_norm_arr;
static float *gate_proj_arr;
static float *up_proj_arr;
static float *down_proj_arr;
static float *temp;
static float *hidden_states_arr;

// Shared memory space across all layers for projection weights
#if DEC_SHARED_MEM
static uint8_t *gate_proj_weight, *down_proj_weight, *up_proj_weight;
static float *gate_scale_ptr, *gate_offset_ptr, *gate_zero_point_ptr;
static float *down_scale_ptr, *down_offset_ptr, *down_zero_point_ptr;
static float *up_scale_ptr, *up_offset_ptr, *up_zero_point_ptr;
#endif

template <typename T>
static void add(Matrix3D<T> a, Matrix3D<T> b, Matrix3D<T> c) {
    PROFILE_START("Int4llamaDecoderLayer::add");
    assert(c.length() == a.length() && a.length() == b.length());

    for (int i = 0; i < a.length(); i++) {
        c.m_data[i] = a.m_data[i] + b.m_data[i];
    }
    PROFILE_END("Int4llamaDecoderLayer::add");
}

inline static float Silu(float x) {
    return x / (1.0f + expf(-x));
}

inline static void SiLuMul(Matrix3D<float> a, Matrix3D<float> b) {
    PROFILE_START("Int4llamaDecoderLayer::MulSiLu");
    for (int i = 0; i < a.length(); i++) {
        // float v = a.m_data[i];
        // float silu_v = v * (1.0 / (1.0 + exp(-1 * v)));
        // a.m_data[i] = silu_v * b.m_data[i];
        a.m_data[i] = Silu(a.m_data[i]) * b.m_data[i];
    }
    PROFILE_END("Int4llamaDecoderLayer::MulSiLu");
}

struct Int4llamaDecoderLayer_output Int4llamaDecoderLayer::forward(std::string param_path, const struct Int4llamaDecoderLayer_input &input, int layer_idx) {
    PROFILE_START(profile_name);

#if DEC_SHARED_MEM
    int x = 1, y = this->hidden_dim, z = this->embed_dim / 2;
    this->gate_proj = Linear_FP_int4(Matrix3D<uint8_t>(gate_proj_weight, x, y, z), param_path + "/gate_proj", 
        Matrix3D<float>(gate_scale_ptr, x, y, z * 2 / QK), Matrix3D<float>(gate_offset_ptr, x, y, z * 2 / QK), 
        Matrix3D<float>(gate_zero_point_ptr, 1, 1, 1));
    y = this->embed_dim, z = this->hidden_dim / 2;
    this->down_proj = Linear_FP_int4(Matrix3D<uint8_t>(down_proj_weight, x, y, z), param_path + "/down_proj", 
        Matrix3D<float>(down_scale_ptr, x, y, z * 2 / QK), Matrix3D<float>(down_offset_ptr, x, y, z * 2 / QK), 
        Matrix3D<float>(down_zero_point_ptr, 1, 1, 1));
    y = this->hidden_dim, z = this->embed_dim / 2;
    this->up_proj = Linear_FP_int4(Matrix3D<uint8_t>(up_proj_weight, x, y, z), param_path + "/up_proj", 
        Matrix3D<float>(up_scale_ptr, x, y, z * 2 / QK), Matrix3D<float>(up_offset_ptr, x, y, z * 2 / QK), 
        Matrix3D<float>(up_zero_point_ptr, 1, 1, 1));
#endif

    // Layernorm
    Matrix3D<float> hidden_states(hidden_states_arr, input.hidden_states.m_dim_x, input.hidden_states.m_dim_y,
                                  input.hidden_states.m_dim_z);
    this->input_layernorm.forward(input.hidden_states, hidden_states, rms_norm_eps);

    // Attention
    struct Int4llamaAttention_input attn_param(hidden_states, input.attention_mask, input.past_key, input.past_value,
                                               input.has_past_key_value, this->layer_idx);
    struct Int4llamaAttention_output attn_output = this->attn.forward(param_path + "/self_attn", attn_param);

    // Residual add
    Matrix3D<float> residual_add(hidden_states_float_arr, input.hidden_states.m_dim_x, input.hidden_states.m_dim_y,
                                 input.hidden_states.m_dim_z);
    add(input.hidden_states, attn_output.attn_output, residual_add);

    // Layernorm
    Matrix3D<float> post_attention_layernorm(final_layer_norm_arr, input.hidden_states.m_dim_x,
                                             input.hidden_states.m_dim_y, input.hidden_states.m_dim_z);
    this->post_attention_layernorm.forward(residual_add, post_attention_layernorm, rms_norm_eps);

    // Gate proj: embedding -> hidden_dim
    Matrix3D<float> gate_proj(gate_proj_arr, input.hidden_states.m_dim_x, input.hidden_states.m_dim_y,
                              this->hidden_dim);
    PROFILE_START("Int4llamaDecoderLayer::gate_proj");
    this->gate_proj.forward(post_attention_layernorm, gate_proj);
    PROFILE_END("Int4llamaDecoderLayer::gate_proj");

    // up proj: embedding -> hidden_dim
    Matrix3D<float> up_proj(up_proj_arr, input.hidden_states.m_dim_x, input.hidden_states.m_dim_y, this->hidden_dim);
    PROFILE_START("Int4llamaDecoderLayer::up_proj");
    this->up_proj.forward(post_attention_layernorm, up_proj);
    PROFILE_END("Int4llamaDecoderLayer::up_proj");

    // silu
    SiLuMul(gate_proj, up_proj);

    // down proj: hidden_dim -> embedding
    Matrix3D<float> down_proj(down_proj_arr, input.hidden_states.m_dim_x, input.hidden_states.m_dim_y, this->embed_dim);
    PROFILE_START("Int4llamaDecoderLayer::down_proj");
    this->down_proj.forward(gate_proj, down_proj);
    PROFILE_END("Int4llamaDecoderLayer::down_proj");

    // Residual add
    add(residual_add, down_proj, residual_add);

    struct Int4llamaDecoderLayer_output output(residual_add, attn_output.attn_probs_reshaped,
                                               attn_output.past_key_value);
    PROFILE_END(profile_name);
    return output;
}

Int4llamaDecoderLayer::Int4llamaDecoderLayer(std::string param_path, const struct model_config config, int layer_idx) {
    if (layer_idx == 0) {
        allocate_aligned_memory(hidden_states_float_arr, config.max_sqlen * config.embed_dim * sizeof(float));
        allocate_aligned_memory(final_layer_norm_arr, config.max_sqlen * config.embed_dim * sizeof(float));
        allocate_aligned_memory(gate_proj_arr, config.max_sqlen * config.hidden_dim * sizeof(float));
        allocate_aligned_memory(up_proj_arr, config.max_sqlen * config.hidden_dim * sizeof(float));
        allocate_aligned_memory(down_proj_arr, config.max_sqlen * config.embed_dim * sizeof(float));
        allocate_aligned_memory(hidden_states_arr, config.max_sqlen * config.embed_dim * sizeof(float));
        Int4llamaAttention::initialized_memory(config);

#if DEC_SHARED_MEM
        // gate_proj
        int gate_weight_length = config.embed_dim * config.hidden_dim * sizeof(uint8_t) / 2;
        allocate_aligned_memory(gate_proj_weight, gate_weight_length);
        allocate_aligned_memory(gate_scale_ptr, (gate_weight_length * 2 * sizeof(float)) / QK);
        allocate_aligned_memory(gate_offset_ptr, (gate_weight_length * 2 * sizeof(float)) / QK);
        allocate_aligned_memory(gate_zero_point_ptr, 1 * sizeof(float));
        // down_proj
        int down_weight_length = config.hidden_dim * config.embed_dim * sizeof(uint8_t) / 2;
        allocate_aligned_memory(down_proj_weight, down_weight_length);
        allocate_aligned_memory(down_scale_ptr, (down_weight_length * 2 * sizeof(float)) / QK);
        allocate_aligned_memory(down_offset_ptr, (down_weight_length * 2 * sizeof(float)) / QK);
        allocate_aligned_memory(down_zero_point_ptr, 1 * sizeof(float));
        // up_proj
        int up_weight_length = config.embed_dim * config.hidden_dim * sizeof(uint8_t) / 2;
        allocate_aligned_memory(up_proj_weight, up_weight_length);
        allocate_aligned_memory(up_scale_ptr, (up_weight_length * 2 * sizeof(float)) / QK);
        allocate_aligned_memory(up_offset_ptr, (up_weight_length * 2 * sizeof(float)) / QK);
        allocate_aligned_memory(up_zero_point_ptr, 1 * sizeof(float));
#endif
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

    this->rms_norm_eps = config.rms_norm_eps;

    this->embed_dim = config.embed_dim;
    this->num_attention_heads = config.num_heads;
    this->hidden_dim = config.hidden_dim;
    this->layer_idx = layer_idx;

    this->attn = Int4llamaAttention(param_path + "/self_attn", config, layer_idx);

#if !(DEC_SHARED_MEM)
    uint8_t *gate_proj_weight, *down_proj_weight, *up_proj_weight;
    allocate_aligned_memory(gate_proj_weight, (config.embed_dim * config.hidden_dim * sizeof(uint8_t)) / 2);
    allocate_aligned_memory(down_proj_weight, (config.hidden_dim * config.embed_dim * sizeof(uint8_t)) / 2);
    allocate_aligned_memory(up_proj_weight, (config.embed_dim * config.hidden_dim * sizeof(uint8_t)) / 2);
    this->gate_proj = Linear_FP_int4(Matrix3D<uint8_t>(gate_proj_weight, 1, config.hidden_dim, config.embed_dim / 2),
                                     (param_path + "/gate_proj"));
    this->down_proj = Linear_FP_int4(Matrix3D<uint8_t>(down_proj_weight, 1, config.embed_dim, config.hidden_dim / 2),
                                     (param_path + "/down_proj"));
    this->up_proj = Linear_FP_int4(Matrix3D<uint8_t>(up_proj_weight, 1, config.hidden_dim, config.embed_dim / 2),
                                   (param_path + "/up_proj"));
#endif
}
