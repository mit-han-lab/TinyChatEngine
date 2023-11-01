#include "Fp32GPTBigCodeForCausalLM.h"

#include <chrono>

#include "operators.h"
#include "utils.h"

Fp32GPTBigCodeForCausalLM::Fp32GPTBigCodeForCausalLM(std::string param_path, const struct model_config config) {
    allocate_aligned_memory(logits_output, config.max_sqlen * config.vocsize * sizeof(float));
    allocate_aligned_memory(lm_head_weight, config.embed_dim * config.vocsize * sizeof(float));

    this->decoder = Fp32GPTBigCodeDecoder(param_path + "/decoder", config);
    this->lm_head =
        Linear_FP(Matrix3D<float>(lm_head_weight, 1, config.vocsize, config.embed_dim), param_path + "/lm_head.bin");
}

struct Fp32GPTBigCodeForCausalLM_output Fp32GPTBigCodeForCausalLM::forward(const struct Fp32GPTBigCodeForCausalLM_input &input) {
    PROFILE_START(profile_name);
    int sqlen = input.input_ids.m_dim_z;

    struct Fp32GPTBigCodeDecoder_output decoder_output;

    if (input.has_past_keys_values) {
        struct Fp32GPTBigCodeDecoder_input decoder_input = {input.input_ids, input.past_keys, input.past_values};
        decoder_output = this->decoder.forward(decoder_input);

    } else {
        struct Fp32GPTBigCodeDecoder_input decoder_input = {input.input_ids};
        decoder_output = this->decoder.forward(decoder_input);
    }

    Matrix3D<float> logits(logits_output, 1, sqlen, this->decoder.voc_size);
    this->lm_head.forward(decoder_output.last_hidden_state, logits);

    struct Fp32GPTBigCodeForCausalLM_output LMoutput = {logits, decoder_output.past_keys, decoder_output.past_values};
    PROFILE_END(profile_name);
    return LMoutput;
}
