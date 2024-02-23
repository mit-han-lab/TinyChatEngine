#include "Int4GPTBigCodeDecoder.h"

#include <cstring>
#include <iostream>

#include "utils.h"

Matrix3D<float> Int4GPTBigCodeDecoder::prepare_decoder_attention_mask(int length, int past_length) {
    PROFILE_START("Int4GPTBigCodeDecoder::prepare_decoder_attention_mask");
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

    PROFILE_END("Int4GPTBigCodeDecoder::prepare_decoder_attention_mask");
    return causal_attention_mask;
}

// Matrix3D<float> Int4GPTBigCodeDecoder::get_position_embed(int sql_length, int past_length) {
//     PROFILE_START("Int4GPTBigCodeDecoder::get_position_embed");
//     const int offset = 2;  // This is specific for GPTBigCode model
//     Matrix3D<float> pos_embeds(pos_embeds_buf, 1, sql_length, this->embed_dim);

//     int start_idx = past_length + offset, end_idx = past_length + sql_length + offset;
//     assert(end_idx < this->embed_positions.lookup.m_dim_y);

//     memcpy(pos_embeds.m_data, &this->embed_positions.lookup.m_data[start_idx * this->embed_dim],
//            sql_length * this->embed_dim * sizeof(float));

//     PROFILE_END("Int4GPTBigCodeDecoder::get_position_embed");
//     return pos_embeds;
// }

Int4GPTBigCodeDecoder::Int4GPTBigCodeDecoder(std::string param_path, const struct model_config config) {
    allocate_aligned_memory(attention_mask_buf, config.max_sqlen * config.max_sqlen * sizeof(float));
    allocate_aligned_memory(pos_embeds_buf, config.max_sqlen * config.embed_dim * sizeof(float));
    allocate_aligned_memory(last_hidden_states_buf, config.max_sqlen * config.embed_dim * sizeof(float));
    allocate_aligned_memory(hidden_states_buf, config.max_sqlen * config.embed_dim * sizeof(float));

    this->voc_size = config.vocsize;
    this->embed_dim = config.embed_dim;
    this->hidden_dim = config.hidden_dim;
    this->num_heads = config.num_heads;
    this->padding_idx = config.padding_idx;
    this->max_position_embeddings = 2048; // To be fixed

    // Embedding
    Matrix3D<float> embweight(new float[voc_size * embed_dim], 1, voc_size, embed_dim);
    this->wte = Embedding(embed_dim, voc_size, padding_idx, embweight);
    load_Embedding_params(this->wte, param_path + "/wte");
    Matrix3D<float> posweight(new float[max_position_embeddings * embed_dim], 1, max_position_embeddings, embed_dim);
    this->wpe = Embedding(embed_dim, max_position_embeddings, padding_idx, posweight);
    load_Embedding_params(this->wpe, param_path + "/wpe");

    // LayerNorm
    Matrix3D<float> LN_weight(new float[config.embed_dim], 1, 1, config.embed_dim);
    Matrix3D<float> LN_bias(new float[config.embed_dim], 1, 1, config.embed_dim);
    struct LayerNorm_params LN_params = {LN_weight, LN_bias};
    this->ln_f = LayerNorm(LN_params);
    load_LayerNorm(this->ln_f, param_path + "/ln_f");

    // Load all the decoder layers
    for (int layer_idx = 0; layer_idx < config.num_layers; layer_idx++) {
        DEBUG_INS(std::cout << "Start loading layer:" << layer_idx << "..." << std::endl;)

        std::string path = param_path + "/layer" + std::to_string(layer_idx);

        Int4GPTBigCodeDecoderLayer layer = Int4GPTBigCodeDecoderLayer(path, config, layer_idx);
        this->layers.push_back(layer);
    }
};

// GPTBigCodeDecoder:
struct Int4GPTBigCodeDecoder_output Int4GPTBigCodeDecoder::forward(const struct Int4GPTBigCodeDecoder_input &input) {
    PROFILE_START(profile_name);
    int sqlen = input.input_ids.m_dim_z, batch_size = input.input_ids.m_dim_x, past_key_values_length = 0;

    // Input token -> Embedding
    // printf(("Int4GPTBigCodeDecoder starts\n");
#ifdef _WIN32
    std::vector<float> inputs_embeds_buf_vec(sqlen * this->embed_dim);
    float *inputs_embeds_buf = &inputs_embeds_buf_vec.front();
#else
    // printf(("uou\n");
    float inputs_embeds_buf[sqlen * this->embed_dim];
    // printf(("wowwww\n");
#endif
    Matrix3D<float> inputs_embeds(inputs_embeds_buf, 1, sqlen, this->embed_dim);
    // printf(("lolll\n");
    this->wte.forward(input.input_ids, inputs_embeds);
    // printf(("aaaaaaa\n");

    if (input.has_past_keys_values) {
        past_key_values_length = input.past_keys[0].m_dim_y;
    }

    // Attention mask  // To be checked
    // printf(("Before prepare_decoder_attention_mask\n");
    Matrix3D<float> causal_attention_mask =
        this->prepare_decoder_attention_mask(sqlen + past_key_values_length, past_key_values_length);

    // Position embeddings
    // printf(("Before get_position_embed\n");
    // Matrix3D<float> pos_embeds = this->get_position_embed(sqlen, past_key_values_length);
#ifdef _WIN32
    std::vector<int> position_ids_buf_vec(sqlen);
    int *position_ids_buf = &position_ids_buf_vec.front();
    std::vector<float> pos_embeds_buf_vec(sqlen * this->embed_dim);
    float *pos_embeds_buf = &pos_embeds_buf_vec.front();
#else
    int position_ids_buf[sqlen];
    float pos_embeds_buf[sqlen * this->embed_dim];
#endif
    Matrix3D<int> position_ids(position_ids_buf, 1, 1, sqlen);
    for (int i = 0; i < sqlen; i++) position_ids.m_data[i] = i + past_key_values_length;
    Matrix3D<float> pos_embeds(pos_embeds_buf, 1, sqlen, this->embed_dim);
    this->wpe.forward(position_ids, pos_embeds);
    
    assert(inputs_embeds.m_dim_x == pos_embeds.m_dim_x);
    assert(inputs_embeds.m_dim_y == pos_embeds.m_dim_y);
    assert(inputs_embeds.m_dim_z == pos_embeds.m_dim_z);
    Matrix3D<float> hidden_states(hidden_states_buf, 1, sqlen, this->embed_dim);
    for (int i = 0; i < inputs_embeds.length(); i++)
        hidden_states.m_data[i] = inputs_embeds.m_data[i] + pos_embeds.m_data[i];

    // Go through each layer
    // printf(("Before layers\n");
    std::vector<Matrix3D<float>> past_keys, past_values;
    for (int i = 0; i < this->layers.size(); i++) {
        if (!input.has_past_keys_values) {
            struct Int4GPTBigCodeDecoderLayer_input l_i = {hidden_states, causal_attention_mask};
            struct Int4GPTBigCodeDecoderLayer_output l_o = this->layers[i].forward(l_i);
            hidden_states = l_o.hidden_states;
            past_keys.push_back(l_o.past_key_value.first);
            past_values.push_back(l_o.past_key_value.second);
        } else {
            struct Int4GPTBigCodeDecoderLayer_input l_i = {hidden_states, causal_attention_mask, input.past_keys[i],
                                                    input.past_values[i]};
            struct Int4GPTBigCodeDecoderLayer_output l_o = this->layers[i].forward(l_i);
            hidden_states = l_o.hidden_states;
            past_keys.push_back(l_o.past_key_value.first);
            past_values.push_back(l_o.past_key_value.second);
        }
    }

    // Layernorm
    Matrix3D<float> last_hidden_states(last_hidden_states_buf, 1, sqlen, this->embed_dim);
    this->ln_f.forward(hidden_states, last_hidden_states);

    struct Int4GPTBigCodeDecoder_output output = {last_hidden_states, past_keys, past_values};
    PROFILE_END(profile_name);
    return output;
}
