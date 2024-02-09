#include "Fp32CLIPVisionTransformer.h"
#include "utils.h"

#include <cstring>
#include <iostream>

Fp32CLIPVisionTransformer::Fp32CLIPVisionTransformer(std::string param_path, const struct model_config config, bool is_vila) {
    allocate_aligned_memory(patch_embeds_buf, 24 * 24 * config.embed_dim * sizeof(float));  // TODO
    allocate_aligned_memory(class_embeds_buf, config.embed_dim * sizeof(float));
    allocate_aligned_memory(pos_embeds_buf, 577 * config.embed_dim * sizeof(float));
    allocate_aligned_memory(last_hidden_states_buf, 576 * config.embed_dim * sizeof(float));
    allocate_aligned_memory(hidden_states_buf, 577 * config.embed_dim * sizeof(float));
    allocate_aligned_memory(embeddings_buf, 577 * config.embed_dim * sizeof(float));
    allocate_aligned_memory(mm_proj_0_arr, 576 * config.hidden_dim * sizeof(float));
    if (is_vila) {
        mm_proj_2_arr = nullptr;
    } else {
        allocate_aligned_memory(mm_proj_2_arr, 576 * config.hidden_dim * sizeof(float));
    }

    this->encoder = Fp32CLIPEncoder(param_path + "/encoder", config);

    this->voc_size = config.vocsize;
    this->embed_dim = config.embed_dim;
    this->hidden_dim = config.hidden_dim;
    this->num_heads = config.num_heads;
    this->padding_idx = config.padding_idx;
    int max_sqlen = config.max_sqlen;

    // Class Embedding
    read_to_array((param_path + "/embeddings/class_embedding/weight.bin").c_str(), class_embeds_buf, config.embed_dim);
    // Patch Embedding
    struct Conv2D_params embed_patch;
    float *patch_weight_buf;
    allocate_aligned_memory(patch_weight_buf, 14 * 14 * 3 * 1024 * sizeof(float));
    Matrix4D<float> patch_weight(patch_weight_buf, 3, 14, 14, 1024);  // TODO
    embed_patch.weight = patch_weight;
    embed_patch.stride_width = 14;
    embed_patch.stride_height = 14;
    this->embed_patch = Conv2D(embed_patch);
    load_Conv2D(this->embed_patch, param_path + "/embeddings/patch_embedding");
    // Position Embedding
    float *posweight_buf;
    allocate_aligned_memory(posweight_buf, 1024 * 577 * sizeof(float));
    Matrix3D<float> posweight(posweight_buf, 1, 577, 1024);  // TODO: num_positions
    this->embed_positions = Embedding(1024, 577, padding_idx, posweight);  // TODO: num_positions
    load_Embedding_params(this->embed_positions, param_path + "/embeddings/position_embedding");

    // LayerNorm
    struct LayerNorm_params pre_layernorm;
    float *pre_layernorm_weight_buf, *pre_layernorm_bias_buf;
    allocate_aligned_memory(pre_layernorm_weight_buf, config.embed_dim * sizeof(float));
    allocate_aligned_memory(pre_layernorm_bias_buf, config.embed_dim * sizeof(float));
    Matrix3D<float> pre_layernorm_weight(pre_layernorm_weight_buf, 1, 1, config.embed_dim);
    Matrix3D<float> pre_layernorm_bias(pre_layernorm_bias_buf, 1, 1, config.embed_dim);
    pre_layernorm.weight = pre_layernorm_weight;
    pre_layernorm.bias = pre_layernorm_bias;
    this->pre_layernorm = LayerNorm(pre_layernorm);
    load_LayerNorm(this->pre_layernorm, param_path + "/pre_layernorm");

    // Projection
    float *mm_proj_0_weight, *mm_proj_2_weight;
    allocate_aligned_memory(mm_proj_0_weight, config.embed_dim * config.hidden_dim * sizeof(float));
    float *mm_proj_0_bias, *mm_proj_2_bias;
    allocate_aligned_memory(mm_proj_0_bias, (config.hidden_dim * sizeof(float)));
    if (is_vila) {
        this->mm_proj_0 = 
            Linear_FP(Matrix3D<float>(mm_proj_0_weight, 1, config.hidden_dim, config.embed_dim), param_path + "/mm_projector/weight.bin",
                      Matrix3D<float>(mm_proj_0_bias, 1, 1, config.hidden_dim), (param_path + "/mm_projector/bias.bin"));
        this->mm_proj_0.has_bias = true;
    } else {
        this->mm_proj_0 = 
            Linear_FP(Matrix3D<float>(mm_proj_0_weight, 1, config.hidden_dim, config.embed_dim), param_path + "/mm_projector_0/weight.bin",
                      Matrix3D<float>(mm_proj_0_bias, 1, 1, config.hidden_dim), (param_path + "/mm_projector_0/bias.bin"));
        this->mm_proj_0.has_bias = true;
        allocate_aligned_memory(mm_proj_2_weight, config.hidden_dim * config.hidden_dim * sizeof(float));
        allocate_aligned_memory(mm_proj_2_bias, (config.hidden_dim * sizeof(float)));
        this->mm_proj_2 = 
            Linear_FP(Matrix3D<float>(mm_proj_2_weight, 1, config.hidden_dim, config.hidden_dim), param_path + "/mm_projector_2/weight.bin",
                      Matrix3D<float>(mm_proj_2_bias, 1, 1, config.hidden_dim), (param_path + "/mm_projector_2/bias.bin"));
        this->mm_proj_2.has_bias = true;
    }
};

// Fp32CLIPVisionTransformer:
struct Fp32CLIPVisionTransformer_output Fp32CLIPVisionTransformer::forward(const struct Fp32CLIPVisionTransformer_input &input, bool is_vila) {
    PROFILE_START(profile_name);
    int sqlen = input.input_image.m_dim_z, batch_size = input.input_image.m_dim_x, past_key_values_length = 0;

    if (input.has_past_keys_values) {
        past_key_values_length = input.past_keys[0].m_dim_y;
    }

    // Attention mask: NULL
    Matrix3D<float> causal_attention_mask;

    // Input image
    Matrix3D<float> input_image(input.input_image.m_data, input.input_image.m_dim_x, input.input_image.m_dim_y,
                                input.input_image.m_dim_z);

    // Patch embeddings
    Matrix3D<float> patch_embeds(patch_embeds_buf, this->embed_dim, 24, 24);  // TODO
    this->embed_patch.forward(input_image, patch_embeds);
    // Class embeddings
    Matrix3D<float> embeddings(embeddings_buf, 1, 577, this->embed_dim);
    Matrix3D<float> class_embeds(class_embeds_buf, 1, 1, this->embed_dim);
    // Concate class embeddings with patch embeddings into embeddings
    memcpy(embeddings.m_data, class_embeds.m_data, class_embeds.length() * sizeof(float));
    memcpy(embeddings.m_data + class_embeds.length(), patch_embeds.m_data, patch_embeds.length() * sizeof(float));
    // Position embeddings
    int position_ids_buf[577];
    Matrix3D<int> position_ids(position_ids_buf, 1, 1, 577);
    for (int i = 0; i < 577; i++) position_ids.m_data[i] = i + past_key_values_length;
    Matrix3D<float> pos_embeds(pos_embeds_buf, 1, 577, this->embed_dim);
    this->embed_positions.forward(position_ids, pos_embeds);

    assert(embeddings.m_dim_x == pos_embeds.m_dim_x);
    assert(embeddings.m_dim_y == pos_embeds.m_dim_y);
    assert(embeddings.m_dim_z == pos_embeds.m_dim_z);
    for (int i = 0; i < embeddings.length(); i++) {
        embeddings.m_data[i] = embeddings.m_data[i] + pos_embeds.m_data[i];
    }

    // Pre-Layernorm
    Matrix3D<float> hidden_states(hidden_states_buf, 1, 577, this->embed_dim);
    this->pre_layernorm.forward(embeddings, hidden_states);

    // CLIP Encoder
    struct Fp32CLIPEncoder_output encoder_output;
    if (input.has_past_keys_values) {
        struct Fp32CLIPEncoder_input encoder_input = {hidden_states, causal_attention_mask, input.past_keys, input.past_values};
        encoder_output = this->encoder.forward(encoder_input);
    } else {
        struct Fp32CLIPEncoder_input encoder_input = {hidden_states, causal_attention_mask};
        encoder_output = this->encoder.forward(encoder_input);
    }

    Matrix3D<float> last_hidden_states(last_hidden_states_buf, 1, 576, this->embed_dim);
    // Copy encoder_output.last_hidden_state[1:] to last_hidden_states
    memcpy(last_hidden_states.m_data, encoder_output.last_hidden_state.m_data + this->embed_dim, 
           last_hidden_states.length() * sizeof(float));

    // Projection 1
    Matrix3D<float> mm_proj_0(mm_proj_0_arr, last_hidden_states.m_dim_x, last_hidden_states.m_dim_y, this->hidden_dim);
    this->mm_proj_0.forward(last_hidden_states, mm_proj_0);
    struct Fp32CLIPVisionTransformer_output output;
    if (is_vila) {
        output = {mm_proj_0, encoder_output.past_keys, encoder_output.past_values};
    } else {
        // GELU
        Gelu(mm_proj_0);
        // LLaVA Projection 2
        Matrix3D<float> mm_proj_2(mm_proj_2_arr, last_hidden_states.m_dim_x, last_hidden_states.m_dim_y, this->hidden_dim);
        this->mm_proj_2.forward(mm_proj_0, mm_proj_2);
        output = {mm_proj_2, encoder_output.past_keys, encoder_output.past_values};
    }

    PROFILE_END(profile_name);
    return output;
}
