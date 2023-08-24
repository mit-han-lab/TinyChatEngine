#include <cstring>
#include <iostream>
#include <cfloat>

#include "Int4llamaDecoder.h"
#include "utils.h"

__global__ void prepare_decoder_attention_mask_half(Matrix3D<float16_t> causal_attention_mask, int length, int past_length) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    
    if(i < length - past_length && j < length) {
        float16_t min = -65504;
        if (i + past_length < j) {
            causal_attention_mask(0, i, j) = min;
        } else {
            causal_attention_mask(0, i, j) = 0;
        }
    }
}

Int4llamaDecoder::Int4llamaDecoder(std::string param_path, const struct model_config config) {
    allocate_aligned_memory_gpu(attention_mask_buf, config.max_sqlen * config.max_sqlen * sizeof(float16_t));
    allocate_aligned_memory_gpu(last_hidden_states_buf, config.max_sqlen * config.embed_dim * sizeof(float16_t));
    allocate_aligned_memory_gpu(hidden_states_buf, config.max_sqlen * config.embed_dim * sizeof(float));
    allocate_aligned_memory_gpu(hidden_states_half_buf, config.max_sqlen * config.embed_dim * sizeof(float16_t));

    this->voc_size = config.vocsize;
    this->embed_dim = config.embed_dim;
    this->hidden_dim = config.hidden_dim;
    this->num_heads = config.num_heads;
    this->padding_idx = config.padding_idx;

    // Embedding
    Matrix3D<float> embweight(new float[voc_size * embed_dim], 1, voc_size, embed_dim);
    this->embed_tokens = Embedding(embed_dim, voc_size, padding_idx, embweight);
    load_Embedding_params(this->embed_tokens, param_path + "/embed_tokens");

    allocate_aligned_memory_gpu(norm_weight_ptr, embed_dim * sizeof(float));
    Matrix3D<float> norm_weight(norm_weight_ptr, 1, 1, embed_dim);
    norm_weight.load((param_path + "/norm/weight.bin").c_str());
    this->norm = LlamaRMSNorm_cuda(norm_weight);

    // Load all the decoder layers
    for (int layer_idx = 0; layer_idx < config.num_layers; layer_idx++) {
        DEBUG_INS(std::cout << "Start loading layer:" << layer_idx << "..." << std::endl;)

        std::string path = param_path + "/layer" + std::to_string(layer_idx);
        Int4llamaDecoderLayer layer = Int4llamaDecoderLayer(path, config, layer_idx);

        this->layers.push_back(layer);
    }
};

// Int4llamaDecoder
struct Int4llamaDecoder_output Int4llamaDecoder::forward(const struct Int4llamaDecoder_input &input) {
    PROFILE_START(profile_name);

    int sqlen = input.input_ids.m_dim_z, past_key_values_length = 0;

    Matrix3D<float> hidden_states_float(hidden_states_buf, 1, sqlen, this->embed_dim);
    this->embed_tokens.forward(input.input_ids, hidden_states_float);

    // Convert from float to float16_t
    Matrix3D<float16_t> hidden_states(hidden_states_half_buf, 1, sqlen, this->embed_dim);
    int threadsPerBlock_1D = 1024;
    int blocksPerGrid =(sqlen * this->embed_dim + threadsPerBlock_1D - 1) / threadsPerBlock_1D;
    float2half<<<blocksPerGrid, threadsPerBlock_1D>>>(hidden_states_buf, hidden_states_half_buf, sqlen * this->embed_dim);

    if (input.has_past_keys_values) {
        past_key_values_length = input.past_keys[0].m_dim_y;
    }

    int length = sqlen + past_key_values_length;
    int past_length = past_key_values_length;
    Matrix3D<float16_t> causal_attention_mask(attention_mask_buf, 1, length - past_length, length);
    dim3 threadsPerBlock(32, 32);
    dim3 numBlocks((length - past_length + threadsPerBlock.x - 1) / threadsPerBlock.x, 
                   (length + threadsPerBlock.y - 1) / threadsPerBlock.y);
    prepare_decoder_attention_mask_half<<<numBlocks, threadsPerBlock>>>(causal_attention_mask, length, past_length);

    std::vector<Matrix3D<float16_t>> past_keys, past_values;
    for (int i = 0; i < this->layers.size(); i++) {
        if (!input.has_past_keys_values) {
            struct Int4llamaDecoderLayer_input l_i = {hidden_states, causal_attention_mask};
            struct Int4llamaDecoderLayer_output l_o = this->layers[i].forward(l_i);

            hidden_states = l_o.hidden_states;
            past_keys.push_back(l_o.past_key_value.first);
            past_values.push_back(l_o.past_key_value.second);
        } else {
            struct Int4llamaDecoderLayer_input l_i = {hidden_states, causal_attention_mask, input.past_keys[i],
                                                      input.past_values[i]};
            struct Int4llamaDecoderLayer_output l_o = this->layers[i].forward(l_i);
            
            hidden_states = l_o.hidden_states;
            past_keys.push_back(l_o.past_key_value.first);
            past_values.push_back(l_o.past_key_value.second);
        }
    }

    Matrix3D<float16_t> last_hidden_states(last_hidden_states_buf, 1, sqlen, this->embed_dim);
    this->norm.forward(hidden_states, last_hidden_states);

    struct Int4llamaDecoder_output output = {last_hidden_states, past_keys, past_values};
    PROFILE_END(profile_name);

    return output;
}

void Int4llamaDecoder::free_cuda_memory() {
    free_aligned_memory_gpu(attention_mask_buf);
    free_aligned_memory_gpu(last_hidden_states_buf);
    free_aligned_memory_gpu(hidden_states_buf);
    free_aligned_memory_gpu(hidden_states_half_buf);
    free_aligned_memory_gpu(norm_weight_ptr);
}
