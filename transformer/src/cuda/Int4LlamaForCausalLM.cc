#include "Int4LlamaForCausalLM.h"

#include <chrono>

#include "operators.h"
#include "utils.h"

Int4LlamaForCausalLM::Int4LlamaForCausalLM(std::string param_path, const struct model_config config) {
    allocate_aligned_memory(logits_output, config.max_sqlen * config.vocsize * sizeof(float));
    allocate_aligned_memory(lm_head_weight, (config.embed_dim * config.vocsize * sizeof(int32_t)) / 8);
    //allocate_aligned_memory(lm_head_weight, config.embed_dim * config.vocsize * sizeof(float));

    this->decoder = Int4llamaDecoder(param_path + "/decoder", config);
    this->lm_head = Linear_FP_int4(Matrix3D<int32_t>(lm_head_weight, 1, config.vocsize / 8, config.embed_dim),
                                   param_path + "/lm_head");
    // this->lm_head =
    //     Linear_FP(Matrix3D<float>(lm_head_weight, 1, config.vocsize, config.embed_dim), param_path + "/lm_head.bin");
}

struct Int4LlamaForCausalLM_output Int4LlamaForCausalLM::forward(const struct Int4LlamaForCausalLM_input &input) {
    // Pycode: Skipped
    // output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    // output_hidden_states = (
    //     output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    // )
    // return_dict = return_dict if return_dict is not None else self.config.use_return_dict
    PROFILE_START(profile_name);
    int sqlen = input.input_ids.m_dim_z;

    // outputs = self.model.decoder(...)
    struct Int4llamaDecoder_output decoder_output;

    if (input.has_past_keys_values) {
        struct Int4llamaDecoder_input decoder_input = {input.input_ids, input.past_keys, input.past_values};
        decoder_output = this->decoder.forward(decoder_input);

    } else {
        struct Int4llamaDecoder_input decoder_input = {input.input_ids};
        decoder_output = this->decoder.forward(decoder_input);
    }

    // logits = self.lm_head(outputs[0]).contiguous()
    Matrix3D<float> logits(logits_output, 1, sqlen, this->decoder.voc_size);
    std::cout << "lm_head" << std::endl;
    this->lm_head.forward(decoder_output.last_hidden_state, logits);
    // print_first_k_elelment("logits_output", logits.m_data, 20);

    struct Int4LlamaForCausalLM_output LMoutput = {logits, decoder_output.past_keys, decoder_output.past_values};
    PROFILE_END(profile_name);
    return LMoutput;
}
