#include "int4llamaDecoderLayer.h"

#include "llama_utils.h"
#include "utils.h"

// Shared memory space across all layers
static float *hidden_states_float_arr;
static float *final_layer_norm_arr;
static float *gate_proj_arr;
static float *up_proj_arr;
static float *down_proj_arr;
static float *temp;
static float *hidden_states_arr;

struct int4llamaDecoderLayer_output int4llamaDecoderLayer::forward(const struct int4llamaDecoderLayer_input &input) {
    PROFILE_START(profile_name);
    Matrix3D<float> hidden_states(hidden_states_arr, input.hidden_states.m_dim_x, input.hidden_states.m_dim_y,
                                  input.hidden_states.m_dim_z);
    this->input_layernorm.forward(input.hidden_states, hidden_states);
    // printf("input_layernorm.sum: %f, weight: %f\n", hidden_states.sum(), this->input_layernorm.weight.sum());
    // print_first_k_elelment("hidden_states", hidden_states.m_data, 20);

    struct int4llamaAttention_input attn_param(hidden_states, input.attention_mask, input.past_key, input.past_value,
                                               input.has_past_key_value, this->layer_idx);
    struct int4llamaAttention_output attn_output = this->attn.forward(attn_param);
    // print_first_k_elelment("attn_output.attn_output", attn_output.attn_output.m_data, 20);
    // read_to_array("assets/tests/OPT_125m/int4llamaAttention_attn_output_len512.bin", attn_output.attn_output.m_data,
    // printf("attn_output.sum: %f\n", attn_output.attn_output.sum());

    // opt.py: residual.add_(hidden_states.to(residual.dtype))
    Matrix3D<float> residual_add(hidden_states_float_arr, input.hidden_states.m_dim_x, input.hidden_states.m_dim_y,
                                 input.hidden_states.m_dim_z);
    add(input.hidden_states, attn_output.attn_output, residual_add);
    // printf("residual_add.sum: %f\n", residual_add.sum());
    // print_first_k_elelment("residual_add", residual_add.m_data, 20);

    // opt.py: hidden_states = self.final_layer_norm(residual)
    Matrix3D<float> post_attention_layernorm(final_layer_norm_arr, input.hidden_states.m_dim_x,
                                             input.hidden_states.m_dim_y, input.hidden_states.m_dim_z);
    this->post_attention_layernorm.forward(residual_add, post_attention_layernorm);
    // printf("post_attention_layernorm.sum: %f\n", post_attention_layernorm.sum());

    // return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
    Matrix3D<float> gate_proj(gate_proj_arr, input.hidden_states.m_dim_x, input.hidden_states.m_dim_y,
                              this->hidden_dim);
    this->gate_proj.forward(post_attention_layernorm, gate_proj);
    // printf("gate_proj.sum: %f, weight: %f\n", gate_proj.sum(), this->gate_proj.weight.sum());
    Matrix3D<float> up_proj(up_proj_arr, input.hidden_states.m_dim_x, input.hidden_states.m_dim_y, this->hidden_dim);
    this->up_proj.forward(post_attention_layernorm, up_proj);
    // printf("up_proj.sum: %f, weight: %f\n", up_proj.sum(), this->up_proj.weight.sum());
    SiLuMul(gate_proj, up_proj);
    Matrix3D<float> down_proj(down_proj_arr, input.hidden_states.m_dim_x, input.hidden_states.m_dim_y, this->embed_dim);
    this->down_proj.forward(gate_proj, down_proj);
    // printf("down_proj.sum: %f, weight: %f\n", down_proj.sum(), this->down_proj.weight.sum());
    // print_first_k_elelment("down_proj", down_proj.m_data, 20);

    add(residual_add, down_proj, residual_add);
    // printf("residual_add: %f\n", residual_add.sum());
    // print_first_k_elelment("residual_add", residual_add.m_data, 20);

    struct int4llamaDecoderLayer_output output(residual_add, attn_output.attn_probs_reshaped,
                                               attn_output.past_key_value);
    PROFILE_END(profile_name);
    return output;
}

int4llamaDecoderLayer::int4llamaDecoderLayer(std::string param_path, const struct model_config config, int layer_idx) {
    if (layer_idx == 0) {
        allocate_aligned_memory(hidden_states_float_arr, config.max_sqlen * config.embed_dim * sizeof(float));
        allocate_aligned_memory(final_layer_norm_arr, config.max_sqlen * config.embed_dim * sizeof(float));
        allocate_aligned_memory(gate_proj_arr, config.max_sqlen * config.hidden_dim * sizeof(float));
        allocate_aligned_memory(up_proj_arr, config.max_sqlen * config.hidden_dim * sizeof(float));
        allocate_aligned_memory(down_proj_arr, config.max_sqlen * config.embed_dim * sizeof(float));
        allocate_aligned_memory(hidden_states_arr, config.max_sqlen * config.embed_dim * sizeof(float));
        int4llamaAttention::initialized_memory(config);
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

    this->attn = int4llamaAttention(param_path + "/self_attn", config);

    int8_t *gate_proj_weight, *down_proj_weight, *up_proj_weight;
    allocate_aligned_memory(gate_proj_weight, (config.embed_dim * config.hidden_dim * sizeof(int8_t)) / 2);
    allocate_aligned_memory(down_proj_weight, (config.hidden_dim * config.embed_dim * sizeof(int8_t)) / 2);
    allocate_aligned_memory(up_proj_weight, (config.embed_dim * config.hidden_dim * sizeof(int8_t)) / 2);
    this->gate_proj = Linear_FP_int4(Matrix3D<int8_t>(gate_proj_weight, 1, config.hidden_dim, config.embed_dim / 2),
                                     (param_path + "/gate_proj/weight.bin"));
    this->down_proj = Linear_FP_int4(Matrix3D<int8_t>(down_proj_weight, 1, config.embed_dim, config.hidden_dim / 2),
                                     (param_path + "/down_proj/weight.bin"));
    this->up_proj = Linear_FP_int4(Matrix3D<int8_t>(up_proj_weight, 1, config.hidden_dim, config.embed_dim / 2),
                                   (param_path + "/up_proj/weight.bin"));
}
