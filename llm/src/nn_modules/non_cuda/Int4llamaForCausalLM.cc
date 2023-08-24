#include "Int4llamaForCausalLM.h"

#include <chrono>

#include "operators.h"
#include "utils.h"

Int4LlamaForCausalLM::Int4LlamaForCausalLM(std::string param_path, const struct model_config config) {
    allocate_aligned_memory(logits_output, config.max_sqlen * config.vocsize * sizeof(float));
    allocate_aligned_memory(lm_head_weight, (config.embed_dim * config.vocsize * sizeof(uint8_t)) / 2);

    this->decoder = Int4llamaDecoder(param_path + "/decoder", config);
    this->lm_head = Linear_FP_int4(Matrix3D<uint8_t>(lm_head_weight, 1, config.vocsize, config.embed_dim / 2),
                                   param_path + "/lm_head");
}

struct Int4LlamaForCausalLM_output Int4LlamaForCausalLM::forward(const struct Int4LlamaForCausalLM_input &input) {
    PROFILE_START(profile_name);
    int sqlen = input.input_ids.m_dim_z;

    struct Int4llamaDecoder_output decoder_output;

    // Call decoder
    if (input.has_past_keys_values) {
        struct Int4llamaDecoder_input decoder_input = {input.input_ids, input.past_keys, input.past_values};
        decoder_output = this->decoder.forward(decoder_input);

    } else {
        struct Int4llamaDecoder_input decoder_input = {input.input_ids};
        decoder_output = this->decoder.forward(decoder_input);
    }

    // Get logits
    Matrix3D<float> logits(logits_output, 1, sqlen, this->decoder.voc_size);
    this->lm_head.forward(decoder_output.last_hidden_state, logits);

    struct Int4LlamaForCausalLM_output LMoutput = {logits, decoder_output.past_keys, decoder_output.past_values};
    PROFILE_END(profile_name);
    return LMoutput;
}
