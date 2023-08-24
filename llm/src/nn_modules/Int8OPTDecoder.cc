#include "Int8OPTDecoder.h"

#include <cstring>
#include <iostream>

#include "utils.h"

Matrix3D<float> Int8OPTDecoder::prepare_decoder_attention_mask(int length, int past_length) {
    PROFILE_START("Int8OPTDecoder::prepare_decoder_attention_mask");
    assert(length - past_length > 0);
    Matrix3D<float> causal_attention_mask(attention_mask_buf, 1, length - past_length, length);
    for (int i = 0; i < length - past_length; i++) {
        for (int j = 0; j < length; j++) {
            if (i + past_length < j) {
                causal_attention_mask(0, i, j) = -65504.0;
            } else {
                causal_attention_mask(0, i, j) = 0.0;
            }
        }
    }

    PROFILE_END("Int8OPTDecoder::prepare_decoder_attention_mask");
    return causal_attention_mask;
}

Matrix3D<float> Int8OPTDecoder::get_position_embed(int sql_length, int past_length) {
    PROFILE_START("Int8OPTDecoder::get_position_embed");
    const int offset = 2;  // This is specific for OPT model
    Matrix3D<float> pos_embeds(pos_embeds_buf, 1, sql_length, this->embed_dim);

    int start_idx = past_length + offset, end_idx = past_length + sql_length + offset;
    assert(end_idx < this->embed_positions.lookup.m_dim_y);

    memcpy(pos_embeds.m_data, &this->embed_positions.lookup.m_data[start_idx * this->embed_dim],
           sql_length * this->embed_dim * sizeof(float));

    PROFILE_END("Int8OPTDecoder::get_position_embed");
    return pos_embeds;
}

Int8OPTDecoder::Int8OPTDecoder(std::string param_path, const struct model_config config) {
    allocate_aligned_memory(attention_mask_buf, config.max_sqlen * config.max_sqlen * sizeof(float));
    allocate_aligned_memory(pos_embeds_buf, config.max_sqlen * config.embed_dim * sizeof(float));
    allocate_aligned_memory(last_hidden_states_buf, config.max_sqlen * config.embed_dim * sizeof(float));
    allocate_aligned_memory(hidden_states_buf, config.max_sqlen * config.embed_dim * sizeof(float));

    this->voc_size = config.vocsize;
    this->embed_dim = config.embed_dim;
    this->hidden_dim = config.hidden_dim;
    this->num_heads = config.num_heads;
    this->padding_idx = config.padding_idx;

    // Embedding
    Matrix3D<float> embweight(new float[voc_size * embed_dim], 1, voc_size, embed_dim);
    this->embed_tokens = Embedding(embed_dim, voc_size, padding_idx, embweight);
    load_Embedding_params(this->embed_tokens, param_path + "/embed_tokens");
    Matrix3D<float> posweight(new float[2048 * embed_dim], 1, 2048, embed_dim);
    this->embed_positions = Embedding(embed_dim, 2048, padding_idx, posweight);
    load_Embedding_params(this->embed_positions, param_path + "/embed_positions");

    // LayerNorm
    Matrix3D<float> LN_weight(new float[embed_dim], 1, 1, embed_dim);
    Matrix3D<float> LN_bias(new float[embed_dim], 1, 1, embed_dim);
    struct LayerNorm_params LN_params = {LN_weight, LN_bias};
    this->final_layer_norm = LayerNorm(LN_params);
    load_LayerNorm(this->final_layer_norm, param_path + "/final_layer_norm");

    // Load all the decoder layers
    for (int layer_idx = 0; layer_idx < config.num_layers; layer_idx++) {
        DEBUG_INS(std::cout << "Start loading layer:" << layer_idx << "..." << std::endl;)
        struct BMM_S8T_S8N_F32T_params qk_bmm;
        struct BMM_S8T_S8N_S8T_params pv_bmm;
        struct W8A8B8O8Linear_params k_proj, v_proj, q_proj;
        Matrix3D<int8_t> k_proj_weight(new int8_t[embed_dim * embed_dim], 1, embed_dim, embed_dim);
        Matrix3D<int8_t> k_proj_bias(new int8_t[embed_dim], 1, 1, embed_dim);
        k_proj.weight = k_proj_weight;
        k_proj.bias = k_proj_bias;
        auto k_proj_op = W8A8B8O8Linear(k_proj);
        Matrix3D<int8_t> v_proj_weight(new int8_t[embed_dim * embed_dim], 1, embed_dim, embed_dim);
        Matrix3D<int8_t> v_proj_bias(new int8_t[embed_dim], 1, 1, embed_dim);
        v_proj.weight = v_proj_weight;
        v_proj.bias = v_proj_bias;
        auto v_proj_op = W8A8B8O8Linear(v_proj);
        Matrix3D<int8_t> q_proj_weight(new int8_t[embed_dim * embed_dim], 1, embed_dim, embed_dim);
        Matrix3D<int8_t> q_proj_bias(new int8_t[embed_dim], 1, 1, embed_dim);
        q_proj.weight = q_proj_weight;
        q_proj.bias = q_proj_bias;
        auto q_proj_op = W8A8B8O8Linear(q_proj);

        struct W8A8BFP32OFP32Linear_params out_proj;
        Matrix3D<int8_t> out_proj_weight(new int8_t[embed_dim * embed_dim], 1, embed_dim, embed_dim);
        Matrix3D<float> out_proj_bias(new float[embed_dim], 1, 1, embed_dim);
        out_proj.weight = out_proj_weight;
        out_proj.bias = out_proj_bias;
        auto out_proj_op = W8A8BFP32OFP32Linear(out_proj);

        struct LayerNormQ_params self_attn_layer_norm, final_layer_norm;
        Matrix3D<float> self_attn_layer_norm_weight(new float[embed_dim], 1, 1, embed_dim);
        Matrix3D<float> self_attn_layer_norm_bias(new float[embed_dim], 1, 1, embed_dim);
        self_attn_layer_norm.weight = self_attn_layer_norm_weight;
        self_attn_layer_norm.bias = self_attn_layer_norm_bias;

        Matrix3D<float> final_layer_norm_weight(new float[embed_dim], 1, 1, embed_dim);
        Matrix3D<float> final_layer_norm_bias(new float[embed_dim], 1, 1, embed_dim);
        final_layer_norm.weight = final_layer_norm_weight;
        final_layer_norm.bias = final_layer_norm_bias;
        LayerNormQ self_attn_layer_norm_op = LayerNormQ(self_attn_layer_norm);
        LayerNormQ final_layer_norm_op = LayerNormQ(final_layer_norm);

        struct W8A8B8O8LinearReLU_params fc1;
        Matrix3D<int8_t> fc1_weight(new int8_t[embed_dim * hidden_dim], 1, hidden_dim, embed_dim);
        Matrix3D<int8_t> fc1_bias_int8(new int8_t[hidden_dim], 1, 1, hidden_dim);
        fc1.weight = fc1_weight;
        fc1.bias_int8 = fc1_bias_int8;
        auto fc1_op = W8A8B8O8LinearReLU(fc1);

        struct W8A8BFP32OFP32Linear_params fc2;
        Matrix3D<int8_t> fc2_weight(new int8_t[embed_dim * hidden_dim], 1, embed_dim, hidden_dim);
        Matrix3D<float> fc2_bias(new float[embed_dim], 1, 1, embed_dim);
        fc2.weight = fc2_weight;
        fc2.bias = fc2_bias;
        auto fc2_op = W8A8BFP32OFP32Linear(fc2);

        auto qk_bmm_op = BMM_S8T_S8N_F32T(qk_bmm);
        auto pv_bmm_op = BMM_S8T_S8N_S8T(pv_bmm);

        std::string path = param_path + "/layer" + std::to_string(layer_idx);

        Int8OPTDecoderLayer layer =
            Int8OPTDecoderLayer(path, config, layer_idx, self_attn_layer_norm_op, final_layer_norm_op, fc1_op, fc2_op,
                                qk_bmm_op, pv_bmm_op, k_proj_op, v_proj_op, q_proj_op, out_proj_op);

        this->layers.push_back(layer);
    }
};

// OPTDecoder:
struct Int8OPTDecoder_output Int8OPTDecoder::forward(const struct Int8OPTDecoder_input &input) {
    PROFILE_START(profile_name);
    int sqlen = input.input_ids.m_dim_z, batch_size = input.input_ids.m_dim_x, past_key_values_length = 0;

    // Input token -> Embedding
    float inputs_embeds_buf[sqlen * this->embed_dim];
    Matrix3D<float> inputs_embeds(inputs_embeds_buf, 1, sqlen, this->embed_dim);
    this->embed_tokens.forward(input.input_ids, inputs_embeds);

    if (input.has_past_keys_values) {
        past_key_values_length = input.past_keys[0].m_dim_y;
    }

    // Attention mask
    Matrix3D<float> causal_attention_mask =
        this->prepare_decoder_attention_mask(sqlen + past_key_values_length, past_key_values_length);

    // Position embeddings
    Matrix3D<float> pos_embeds = this->get_position_embed(sqlen, past_key_values_length);
    assert(inputs_embeds.m_dim_x == pos_embeds.m_dim_x);
    assert(inputs_embeds.m_dim_y == pos_embeds.m_dim_y);
    assert(inputs_embeds.m_dim_z == pos_embeds.m_dim_z);
    Matrix3D<float> hidden_states(hidden_states_buf, 1, sqlen, this->embed_dim);
    for (int i = 0; i < inputs_embeds.length(); i++)
        hidden_states.m_data[i] = inputs_embeds.m_data[i] + pos_embeds.m_data[i];

    // Go through each layer
    std::vector<Matrix3D<int8_t>> past_keys, past_values;
    for (int i = 0; i < this->layers.size(); i++) {
        if (!input.has_past_keys_values) {
            struct Int8OPTDecoderLayer_input l_i = {hidden_states, causal_attention_mask};
            struct Int8OPTDecoderLayer_output l_o = this->layers[i].forward(l_i);
            hidden_states = l_o.hidden_states;
            past_keys.push_back(l_o.past_key_value.first);
            past_values.push_back(l_o.past_key_value.second);
        } else {
            struct Int8OPTDecoderLayer_input l_i = {hidden_states, causal_attention_mask, input.past_keys[i],
                                                    input.past_values[i]};
            struct Int8OPTDecoderLayer_output l_o = this->layers[i].forward(l_i);
            hidden_states = l_o.hidden_states;
            past_keys.push_back(l_o.past_key_value.first);
            past_values.push_back(l_o.past_key_value.second);
        }
    }

    // Layernorm
    Matrix3D<float> last_hidden_states(last_hidden_states_buf, 1, sqlen, this->embed_dim);
    this->final_layer_norm.forward(hidden_states, last_hidden_states);

    struct Int8OPTDecoder_output output = {last_hidden_states, past_keys, past_values};
    PROFILE_END(profile_name);
    return output;
}
