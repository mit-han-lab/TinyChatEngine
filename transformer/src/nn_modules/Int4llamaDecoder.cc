#include "Int4llamaDecoder.h"

#include <cstring>
#include <iostream>

#include "utils.h"

Matrix3D<float> Int4llamaDecoder::prepare_decoder_attention_mask(int length, int past_length) {
    PROFILE_START("Int4llamaDecoder::prepare_decoder_attention_mask");
    assert(length - past_length > 0);
    Matrix3D<float> causal_attention_mask(attention_mask_buf, 1, length - past_length, length);
    float min = std::numeric_limits<float>::lowest();
    for (int i = 0; i < length - past_length; i++) {
        for (int j = 0; j < length; j++) {
            if (i + past_length < j) {
                causal_attention_mask(0, i, j) = min;
            } else {
                causal_attention_mask(0, i, j) = 0.0;
            }
        }
    }

    PROFILE_END("Int4llamaDecoder::prepare_decoder_attention_mask");
    return causal_attention_mask;
}

Int4llamaDecoder::Int4llamaDecoder(std::string param_path, const struct model_config config) {
    allocate_aligned_memory(attention_mask_buf, config.max_sqlen * config.max_sqlen * sizeof(float));
    allocate_aligned_memory(pos_embeds_buf, config.max_sqlen * config.embed_dim * sizeof(float));
    allocate_aligned_memory(last_hidden_states_buf, config.max_sqlen * config.embed_dim * sizeof(float));
    allocate_aligned_memory(hidden_states_buf, config.max_sqlen * config.embed_dim * sizeof(float));

    this->voc_size = config.vocsize;
    this->embed_dim = config.embed_dim;
    this->hidden_dim = config.hidden_dim;
    this->num_heads = config.num_heads;
    this->padding_idx = config.padding_idx;

    int max_sqlen = config.max_sqlen;

    // Embedding
    Matrix3D<float> embweight(new float[voc_size * embed_dim], 1, voc_size, embed_dim);
    this->embed_tokens = Embedding(embed_dim, voc_size, padding_idx, embweight);
    load_Embedding_params(this->embed_tokens, param_path + "/embed_tokens");

    // Norm
    Matrix3D<float> norm_weight(new float[embed_dim], 1, 1, embed_dim);
    norm_weight.load((param_path + "/norm/weight.bin").c_str());
    this->norm = LlamaRMSNorm(norm_weight);

    // Load all the decoder layers
    for (int layer_idx = 0; layer_idx < config.num_layers; layer_idx++) {
        DEBUG_INS(std::cout << "Start loading layer:" << layer_idx << "..." << std::endl;)

        std::string path = param_path + "/layer" + std::to_string(layer_idx);
        Int4llamaDecoderLayer layer = Int4llamaDecoderLayer(path, config, layer_idx);

        this->layers.push_back(layer);
    }
};

// Int4llamaDecoder:
struct Int4llamaDecoder_output Int4llamaDecoder::forward(const struct Int4llamaDecoder_input &input) {
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

    // Go through each layer
    Matrix3D<float> hidden_states = inputs_embeds;
    std::vector<Matrix3D<float>> past_keys, past_values;
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

    // Layernorm
    Matrix3D<float> last_hidden_states(last_hidden_states_buf, 1, sqlen, this->embed_dim);
    this->norm.forward(hidden_states, last_hidden_states);

    struct Int4llamaDecoder_output output = {last_hidden_states, past_keys, past_values};
    PROFILE_END(profile_name);
    return output;
}
