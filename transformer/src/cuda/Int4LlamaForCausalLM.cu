#include "Int4LlamaForCausalLM.h"

#include <chrono>

#include "operators.h"
#include "utils.h"

Int4LlamaForCausalLM::Int4LlamaForCausalLM(std::string param_path, const struct model_config config) {
    allocate_aligned_memory_gpu(logits_output, config.max_sqlen * config.vocsize * sizeof(float));
    allocate_aligned_memory_gpu(lm_head_weight, (config.embed_dim * config.vocsize * sizeof(int)) / 8);
    //allocate_aligned_memory_gpu(lm_head_weight, config.embed_dim * config.vocsize * sizeof(float));

    this->decoder = Int4llamaDecoder(param_path + "/decoder", config);
    this->lm_head = Linear_half_int4_ref(Matrix3D<int>(lm_head_weight, 1, config.vocsize / 8, config.embed_dim),
                                   param_path + "/lm_head");
    // this->lm_head =
    //     Linear_FP(Matrix3D<float>(lm_head_weight, 1, config.vocsize, config.embed_dim), param_path + "/lm_head.bin");
}

struct Int4LlamaForCausalLM_output Int4LlamaForCausalLM::forward(const struct Int4LlamaForCausalLM_input &input) {
    PROFILE_START(profile_name);
    int sqlen = input.input_ids.m_dim_z;

    struct Int4llamaDecoder_output decoder_output;

    if (input.has_past_keys_values) {
        struct Int4llamaDecoder_input decoder_input = {input.input_ids, input.past_keys, input.past_values};
        decoder_output = this->decoder.forward(decoder_input);

    } else {
        struct Int4llamaDecoder_input decoder_input = {input.input_ids};
        decoder_output = this->decoder.forward(decoder_input);
    }

    Matrix3D<float> logits(logits_output, 1, sqlen, this->decoder.voc_size);
    this->lm_head.forward(decoder_output.last_hidden_state, logits);
    cudaDeviceSynchronize();

    struct Int4LlamaForCausalLM_output LMoutput = {logits, decoder_output.past_keys, decoder_output.past_values};
    PROFILE_END(profile_name);

    // cudaFree(lm_head_weight);

    return LMoutput;
}