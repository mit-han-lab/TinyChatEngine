#include "Fp32llamaDecoderLayer.h"

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
    PROFILE_START("Fp32llamaDecoderLayer::add");
    assert(c.length() == a.length() && a.length() == b.length());

    for (int i = 0; i < a.length(); i++) {
        c.m_data[i] = a.m_data[i] + b.m_data[i];
    }
    PROFILE_END("Fp32llamaDecoderLayer::add");
}

// struct Fp32llamaDecoderLayer_output Fp32llamaDecoderLayer::forward(const struct Fp32llamaDecoderLayer_input &input)
// {
//   PROFILE_START(profile_name);
//   Matrix3D<float> hidden_states(hidden_states_arr, input.hidden_states.m_dim_x,
//                                      input.hidden_states.m_dim_y, input.hidden_states.m_dim_z);
//   this->input_layernorm.forward(input.hidden_states, hidden_states);
//   // print_first_k_elelment("hidden_states", hidden_states.m_data, 20);

//   struct Fp32llamaAttention_input attn_param(hidden_states, input.attention_mask, input.past_key, input.past_value,
//                                            input.has_past_key_value, this->layer_idx);
//   struct Fp32llamaAttention_output attn_output = this->attn.forward(attn_param);
//   // print_first_k_elelment("attn_output.attn_output", attn_output.attn_output.m_data, 20);
//   // read_to_array("assets/tests/OPT_125m/Fp32llamaAttention_attn_output_len512.bin", attn_output.attn_output.m_data,
//   // attn_output.attn_output.length());

//   // opt.py: residual.add_(hidden_states.to(residual.dtype))
//   Matrix3D<float> residual_add(hidden_states_float_arr, input.hidden_states.m_dim_x, input.hidden_states.m_dim_y,
//                                input.hidden_states.m_dim_z);
//   add(input.hidden_states, attn_output.attn_output, residual_add);
//   // print_first_k_elelment("residual_add", residual_add.m_data, 20);

//   // opt.py: hidden_states = self.final_layer_norm(residual)
//   Matrix3D<float> final_layer_norm(final_layer_norm_arr, input.hidden_states.m_dim_x, input.hidden_states.m_dim_y,
//                                    input.hidden_states.m_dim_z);
//   this->final_layer_norm.forward(residual_add, final_layer_norm);
//   // read_to_array("assets/tests/OPT_1.3B/layer23_final_layer_norm.bin", final_layer_norm.m_data,
//   // final_layer_norm.length()); print_first_k_elelment("final_layer_norm", final_layer_norm.m_data, 20);

//   // self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
//   // self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
//   // self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
//   // self.act_fn = ACT2FN[hidden_act]

//   // opt.py: hidden_states = self.fc1(hidden_states)
//   Matrix3D<float> fc1_out(fc_1_arr, input.hidden_states.m_dim_x, input.hidden_states.m_dim_y, this->hidden_dim);
//   // this->fc1.x = final_layer_norm;
//   // this->fc1.output = fc1_out;
//   this->fc1.forward(final_layer_norm, fc1_out);
//   // print_first_k_elelment("fc1_out", fc1_out.m_data, 20);

//   // opt.py: hidden_states = self.fc2(hidden_states)
//   Matrix3D<float> fc2_out(fc_2_arr, input.hidden_states.m_dim_x, input.hidden_states.m_dim_y,
//                           input.hidden_states.m_dim_z);
//   this->fc2.forward(fc1_out, fc2_out);
//   // read_to_array("assets/tests/OPT_1.3B/fc2_out.bin", fc2_out.m_data, fc2_out.length());
//   // print_first_k_elelment("fc2_out", fc2_out.m_data, 20);

//   // opt.py: residual.add_(hidden_states.to(residual.dtype))
//   add(residual_add, fc2_out, residual_add);
//   // print_first_k_elelment("residual_add", residual_add.m_data, 20);

//   struct Fp32llamaDecoderLayer_output output(residual_add, attn_output.attn_probs_reshaped,
//   attn_output.past_key_value); PROFILE_END(profile_name); return output;
// }

Fp32llamaDecoderLayer::Fp32llamaDecoderLayer(std::string param_path, const struct model_config config, int layer_idx) {
    if (layer_idx == 0) {
        allocate_aligned_memory(hidden_states_float_arr, config.max_sqlen * config.embed_dim * sizeof(float));
        allocate_aligned_memory(final_layer_norm_arr, config.max_sqlen * config.embed_dim * sizeof(float));
        // allocate_aligned_memory(fc_1_arr, config.max_sqlen * config.hidden_dim * sizeof(float));
        // allocate_aligned_memory(fc_2_arr, config.max_sqlen * config.embed_dim * sizeof(float));
        allocate_aligned_memory(hidden_states_arr, config.max_sqlen * config.embed_dim * sizeof(float));
        Fp32llamaAttention::initialized_memory(config);
    }

    float *input_layernorm_weight_ptr;
    allocate_aligned_memory(input_layernorm_weight_ptr, config.embed_dim * sizeof(float));
    Matrix3D<float> input_layernorm_weight(input_layernorm_weight_ptr, 1, 1, config.embed_dim);
    input_layernorm_weight.load((param_path + "/input_layernorm_weight/weight.bin").c_str());
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
}
