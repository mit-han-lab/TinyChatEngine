#include "Int4llamaDecoderLayer.h"
#include "utils.h"

// Shared memory space across all layers
static float16_t *hidden_states_half_arr = nullptr;
static float16_t *final_layer_norm_arr = nullptr;
static float16_t *gate_proj_arr = nullptr;
static float16_t *up_proj_arr = nullptr;
static float16_t *down_proj_arr = nullptr;
static float16_t *hidden_states_arr = nullptr;

void add_half(Matrix3D<float16_t> a, Matrix3D<float16_t> b, Matrix3D<float16_t> c, int num_heads){
    struct metal_params params;
    params.A.half_data_ptr = a.m_data;
    params.B.half_data_ptr = b.m_data;
    params.C.half_data_ptr = c.m_data;
    params.num_heads = num_heads;
    params.sqlen = a.length();
    params.op = METAL_KERNEL_ADD_HALF;
    add_node(&params);
}

void SiLuMul_half(Matrix3D<float16_t> gate_proj, Matrix3D<float16_t> up_proj){
    struct metal_params params;

    params.A.half_data_ptr = gate_proj.m_data;
    params.B.half_data_ptr = up_proj.m_data;
    params.sqlen = gate_proj.length();
    params.op = METAL_KERNEL_SILUMUL_HALF;
    add_node(&params);
}

Int4llamaDecoderLayer::Int4llamaDecoderLayer(std::string param_path, const struct model_config config, int layer_idx) {
    if (layer_idx == 0) {
        allocate_aligned_memory(hidden_states_half_arr, config.max_sqlen * config.embed_dim * sizeof(float16_t));
        allocate_aligned_memory(final_layer_norm_arr, config.max_sqlen * config.embed_dim * sizeof(float16_t));
        allocate_aligned_memory(gate_proj_arr, config.max_sqlen * config.hidden_dim * sizeof(float16_t));
        allocate_aligned_memory(up_proj_arr, config.max_sqlen * config.hidden_dim * sizeof(float16_t));
        allocate_aligned_memory(down_proj_arr, config.max_sqlen * config.embed_dim * sizeof(float16_t));
        allocate_aligned_memory(hidden_states_arr, config.max_sqlen * config.embed_dim * sizeof(float16_t));
        Int4llamaAttention::initialized_memory(config);
    }

    allocate_aligned_memory(input_layernorm_weight_ptr, config.embed_dim * sizeof(float));
    Matrix3D<float> input_layernorm_weight(input_layernorm_weight_ptr, 1, 1, config.embed_dim);
    input_layernorm_weight.load((param_path + "/input_layernorm/weight.bin").c_str());
    // METAL: metal rms_norm
    this->input_layernorm = LlamaRMSNorm_metal(input_layernorm_weight);

    allocate_aligned_memory(post_attention_layernorm_ptr, config.embed_dim * sizeof(float));
    Matrix3D<float> post_attention_layernorm_weight(post_attention_layernorm_ptr, 1, 1, config.embed_dim);
    post_attention_layernorm_weight.load((param_path + "/post_attention_layernorm/weight.bin").c_str());
    // METAL: metal rms_norm
    this->post_attention_layernorm = LlamaRMSNorm_metal(post_attention_layernorm_weight);

    this->rms_norm_eps = config.rms_norm_eps;

    this->embed_dim = config.embed_dim;
    this->num_attention_heads = config.num_heads;
    this->hidden_dim = config.hidden_dim;
    this->layer_idx = layer_idx;

    this->attn = Int4llamaAttention(param_path + "/self_attn", config, layer_idx);

    allocate_aligned_memory(gate_proj_weight, (config.embed_dim * config.hidden_dim * sizeof(int)) / 8);
    allocate_aligned_memory(down_proj_weight, (config.hidden_dim * config.embed_dim * sizeof(int)) / 8 + 1);
    allocate_aligned_memory(up_proj_weight, (config.embed_dim * config.hidden_dim * sizeof(int)) / 8);
    this->gate_proj = Linear_half_int4(Matrix3D<int>(gate_proj_weight, 1, config.hidden_dim, config.embed_dim / 8),
                                     (param_path + "/gate_proj"));
    this->down_proj = Linear_half_int4(Matrix3D<int>(down_proj_weight, 1, config.embed_dim, config.hidden_dim / 8),
                                     (param_path + "/down_proj"));
    this->up_proj = Linear_half_int4(Matrix3D<int>(up_proj_weight, 1, config.hidden_dim, config.embed_dim / 8),
                                   (param_path + "/up_proj"));
}

struct Int4llamaDecoderLayer_output Int4llamaDecoderLayer::forward(std::string param_path, const struct Int4llamaDecoderLayer_input &input, int layer_idx) {
    PROFILE_START(profile_name);

    Matrix3D<float16_t> hidden_states(hidden_states_arr, input.hidden_states.m_dim_x, input.hidden_states.m_dim_y,
                                  input.hidden_states.m_dim_z);
    this->input_layernorm.forward(input.hidden_states, hidden_states, rms_norm_eps);

    struct Int4llamaAttention_input attn_param(hidden_states, input.attention_mask, input.past_key, input.past_value,
                                               input.has_past_key_value, this->layer_idx);
    struct Int4llamaAttention_output attn_output = this->attn.forward(param_path + "/self_attn", attn_param);

    Matrix3D<float16_t> residual_add(hidden_states_half_arr, input.hidden_states.m_dim_x, input.hidden_states.m_dim_y,
                                 input.hidden_states.m_dim_z);
    // int threadsPerBlock = 1024;
    // int blocksPerGrid =(input.hidden_states.length() + threadsPerBlock - 1) / threadsPerBlock;
    // METAL: add interface
    add_half(input.hidden_states, attn_output.attn_output, residual_add, this->num_attention_heads);

    Matrix3D<float16_t> post_attention_layernorm(final_layer_norm_arr, input.hidden_states.m_dim_x,
                                             input.hidden_states.m_dim_y, input.hidden_states.m_dim_z);
    this->post_attention_layernorm.forward(residual_add, post_attention_layernorm, rms_norm_eps);

    Matrix3D<float16_t> gate_proj(gate_proj_arr, input.hidden_states.m_dim_x, input.hidden_states.m_dim_y,
                              this->hidden_dim);
    this->gate_proj.forward(post_attention_layernorm, gate_proj);

    Matrix3D<float16_t> up_proj(up_proj_arr, input.hidden_states.m_dim_x, input.hidden_states.m_dim_y, this->hidden_dim);
    this->up_proj.forward(post_attention_layernorm, up_proj);

    // int blocksPerGrid2 =(gate_proj.length() + threadsPerBlock - 1) / threadsPerBlock;
    // METAL: add interface
    SiLuMul_half(gate_proj, up_proj);

    Matrix3D<float16_t> down_proj(down_proj_arr, input.hidden_states.m_dim_x, input.hidden_states.m_dim_y, this->embed_dim);
    this->down_proj.forward(gate_proj, down_proj);

    // int blocksPerGrid3 =(residual_add.length() + threadsPerBlock - 1) / tchreadsPerBlock;
    // METAL: add interface
    add_half(residual_add, down_proj, residual_add, this->num_attention_heads);

    struct Int4llamaDecoderLayer_output output(residual_add, attn_output.attn_probs_reshaped,
                                               attn_output.past_key_value);
    PROFILE_END(profile_name);

    return output;
}
