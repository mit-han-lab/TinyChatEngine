#include <cstring>
#include <iostream>
#include <cfloat>

#include "Int4llamaDecoder.h"
#include "utils.h"

void prepare_decoder_attention_mask_half(Matrix3D<float16_t> causal_attention_mask, int length, int past_length){
    struct metal_params params;
    params.A.half_data_ptr = causal_attention_mask.m_data;
    params.sqlen = length;
    params.past_sqlen = past_length;
    params.op = METAL_KERNEL_PREPARE_DECODER_ATTENTION_MASK_HALF;
    add_node(&params);
}

void float2half(Matrix3D<float> hidden_states_buf, Matrix3D<float16_t> hidden_states_half_buf, int sq_embed){
    struct metal_params params;
    params.A.data_ptr = hidden_states_buf.m_data;
    params.B.half_data_ptr = hidden_states_half_buf.m_data;
    params.sqlen = sq_embed;
    params.op = METAL_KERNEL_FLOAT2HALF;
    add_node(&params);
}


Int4llamaDecoder::Int4llamaDecoder(std::string param_path, const struct model_config config) {
    allocate_aligned_memory(attention_mask_buf, config.max_sqlen * config.max_sqlen * sizeof(float16_t));
    allocate_aligned_memory(last_hidden_states_buf, config.max_sqlen * config.embed_dim * sizeof(float16_t));
    allocate_aligned_memory(hidden_states_buf, config.max_sqlen * config.embed_dim * sizeof(float));
    allocate_aligned_memory(hidden_states_half_buf, config.max_sqlen * config.embed_dim * sizeof(float16_t));

    this->voc_size = config.vocsize;
    this->embed_dim = config.embed_dim;
    this->hidden_dim = config.hidden_dim;
    this->num_heads = config.num_heads;
    this->padding_idx = config.padding_idx;
    this->rms_norm_eps = config.rms_norm_eps;

    // Embedding
    Matrix3D<float> embweight(new float[voc_size * embed_dim], 1, voc_size, embed_dim);
    // METAL: Metal Embedding
    this->embed_tokens = Embedding(embed_dim, voc_size, padding_idx, embweight); // METAL
    load_Embedding_params(this->embed_tokens, param_path + "/embed_tokens");

    allocate_aligned_memory(norm_weight_ptr, embed_dim * sizeof(float));
    Matrix3D<float> norm_weight(norm_weight_ptr, 1, 1, embed_dim);
    norm_weight.load((param_path + "/norm/weight.bin").c_str());
    // METAL: Metal RMSNorm
    this->norm = LlamaRMSNorm_metal(norm_weight);

    // Load all the decoder layers
    for (int layer_idx = 0; layer_idx < config.num_layers; layer_idx++) {
        DEBUG_INS(std::cout << "Start loading layer:" << layer_idx << "..." << std::endl;)

        std::string path = param_path + "/layer" + std::to_string(layer_idx);
        Int4llamaDecoderLayer layer = Int4llamaDecoderLayer(path, config, layer_idx);

        this->layers.push_back(layer);
    }
};

// Int4llamaDecoder
struct Int4llamaDecoder_output Int4llamaDecoder::forward(std::string param_path, const struct Int4llamaDecoder_input &input) {
    PROFILE_START(profile_name);

    int sqlen = input.input_ids.m_dim_z, past_key_values_length = 0;

    Matrix3D<float> hidden_states_float(hidden_states_buf, 1, sqlen, this->embed_dim);
    this->embed_tokens.forward(input.input_ids, hidden_states_float);

    // Convert from float to float16_t
    Matrix3D<float16_t> hidden_states(hidden_states_half_buf, 1, sqlen, this->embed_dim);
    int threadsPerBlock_1D = 1024;
    int blocksPerGrid =(sqlen * this->embed_dim + threadsPerBlock_1D - 1) / threadsPerBlock_1D;
    // METAL: more kernels
    float2half(hidden_states_float, hidden_states, sqlen * this->embed_dim);

    if (input.has_past_keys_values) {
        past_key_values_length = input.past_keys[0].m_dim_y;
    }

    int length = sqlen + past_key_values_length;
    int past_length = past_key_values_length;
    Matrix3D<float16_t> causal_attention_mask(attention_mask_buf, 1, length - past_length, length);
    // METAL: more kernels
    prepare_decoder_attention_mask_half(causal_attention_mask, length, past_length);

    std::vector<Matrix3D<float16_t>> past_keys, past_values;
    for (int i = 0; i < this->layers.size(); i++) {
        std::string path = param_path + "/layer" + std::to_string(i);

        if (!input.has_past_keys_values) {
            struct Int4llamaDecoderLayer_input l_i = {hidden_states, causal_attention_mask};
            struct Int4llamaDecoderLayer_output l_o = this->layers[i].forward(path, l_i, i);

            hidden_states = l_o.hidden_states;
            past_keys.push_back(l_o.past_key_value.first);
            past_values.push_back(l_o.past_key_value.second);
        } else {
            struct Int4llamaDecoderLayer_input l_i = {hidden_states, causal_attention_mask, input.past_keys[i],
                                                      input.past_values[i]};
            struct Int4llamaDecoderLayer_output l_o = this->layers[i].forward(path, l_i, i);

            hidden_states = l_o.hidden_states;
            past_keys.push_back(l_o.past_key_value.first);
            past_values.push_back(l_o.past_key_value.second);
        }
    }

    Matrix3D<float16_t> last_hidden_states(last_hidden_states_buf, 1, sqlen, this->embed_dim);
    this->norm.forward(hidden_states, last_hidden_states, rms_norm_eps);

    struct Int4llamaDecoder_output output = {last_hidden_states, past_keys, past_values};
    PROFILE_END(profile_name);

    return output;
}