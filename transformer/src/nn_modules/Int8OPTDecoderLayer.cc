#include "Int8OPTDecoderLayer.h"

#include "utils.h"

// Shared memory space across all layers
static float *hidden_states_float_arr;
static int8_t *final_layer_norm_arr;
static int8_t *fc_1_arr;
static float *fc_2_arr;
static float *temp;
static int8_t *hidden_states_int8_arr;

template <typename T>
void add(Matrix3D<T> a, Matrix3D<T> b, Matrix3D<T> c) {
    PROFILE_START("Int8OPTDecoderLayer::add");
    assert(c.length() == a.length() && a.length() == b.length());

    for (int i = 0; i < a.length(); i++) {
        c.m_data[i] = a.m_data[i] + b.m_data[i];
    }
    PROFILE_END("Int8OPTDecoderLayer::add");
}

struct Int8OPTDecoderLayer_output Int8OPTDecoderLayer::forward(const struct Int8OPTDecoderLayer_input &input) {
    PROFILE_START(profile_name);
    // Layernorm
    Matrix3D<int8_t> hidden_states_int8(hidden_states_int8_arr, input.hidden_states.m_dim_x,
                                        input.hidden_states.m_dim_y, input.hidden_states.m_dim_z);
    this->self_attn_layer_norm.forward(input.hidden_states, hidden_states_int8);

    // Attention
    struct Int8OPTAttention_input attn_param(hidden_states_int8, input.attention_mask, input.past_key, input.past_value,
                                             input.has_past_key_value, this->layer_idx);
    struct Int8OPTAttention_output attn_output = this->attn.forward(attn_param);

    // Residual add
    Matrix3D<float> residual_add(hidden_states_float_arr, input.hidden_states.m_dim_x, input.hidden_states.m_dim_y,
                                 input.hidden_states.m_dim_z);
    add(input.hidden_states, attn_output.attn_output, residual_add);

    // Layernorm
    Matrix3D<int8_t> final_layer_norm(final_layer_norm_arr, input.hidden_states.m_dim_x, input.hidden_states.m_dim_y,
                                      input.hidden_states.m_dim_z);
    this->final_layer_norm.forward(residual_add, final_layer_norm);

    // FC
    Matrix3D<int8_t> fc1_out(fc_1_arr, input.hidden_states.m_dim_x, input.hidden_states.m_dim_y, this->hidden_dim);
    this->fc1.forward(final_layer_norm, fc1_out);
    Matrix3D<float> fc2_out(fc_2_arr, input.hidden_states.m_dim_x, input.hidden_states.m_dim_y,
                            input.hidden_states.m_dim_z);
    this->fc2.forward(fc1_out, fc2_out);

    // Reidual add
    add(residual_add, fc2_out, residual_add);

    struct Int8OPTDecoderLayer_output output(residual_add, attn_output.attn_probs_reshaped, attn_output.past_key_value);
    PROFILE_END(profile_name);
    return output;
}

Int8OPTDecoderLayer::Int8OPTDecoderLayer(std::string param_path, const struct model_config config, int layer_idx,
                                         LayerNormQ self_attn_layer_norm, LayerNormQ final_layer_norm,
                                         W8A8B8O8LinearReLU fc1, W8A8BFP32OFP32Linear fc2, BMM_S8T_S8N_F32T qk_bmm,
                                         BMM_S8T_S8N_S8T pv_bmm, W8A8B8O8Linear k_proj, W8A8B8O8Linear v_proj,
                                         W8A8B8O8Linear q_proj, W8A8BFP32OFP32Linear out_proj) {
    if (layer_idx == 0) {
        allocate_aligned_memory(hidden_states_float_arr, config.max_sqlen * config.embed_dim * sizeof(float));
        allocate_aligned_memory(final_layer_norm_arr, config.max_sqlen * config.embed_dim * sizeof(int8_t));
        allocate_aligned_memory(fc_1_arr, config.max_sqlen * config.hidden_dim * sizeof(int8_t));
        allocate_aligned_memory(fc_2_arr, config.max_sqlen * config.embed_dim * sizeof(float));
        allocate_aligned_memory(hidden_states_int8_arr, config.max_sqlen * config.embed_dim * sizeof(int8_t));
        Int8OPTAttention::initialized_memory(config);
    }

    load_LayerNormQ(self_attn_layer_norm, param_path + "/self_attn_layer_norm");
    load_W8A8B8O8LinearReLU_params(fc1, param_path + "/fc1");
    load_W8A8BFP32OFP32Linear_params(fc2, param_path + "/fc2");
    load_LayerNormQ(final_layer_norm, param_path + "/final_layer_norm");

    this->embed_dim = config.embed_dim;
    this->num_attention_heads = config.num_heads;
    this->hidden_dim = config.hidden_dim;
    this->layer_idx = layer_idx;
    this->self_attn_layer_norm = self_attn_layer_norm;
    this->fc1 = fc1;
    this->fc2 = fc2;
    this->final_layer_norm = final_layer_norm;

    this->attn = Int8OPTAttention(param_path + "/self_attn", config, qk_bmm, pv_bmm, k_proj, v_proj, q_proj, out_proj);
}
