#include "Int8OPTDecoder.h"

struct OPTForCausalLM_output {
    Matrix3D<float> logits;
    std::vector<Matrix3D<int8_t>> past_keys, past_values;
};
struct OPTForCausalLM_input {
    Matrix3D<int> input_ids;
    std::vector<Matrix3D<int8_t>> past_keys, past_values;
    bool has_past_keys_values;

    OPTForCausalLM_input(Matrix3D<int> input_ids_) : input_ids(input_ids_) { has_past_keys_values = false; }
    OPTForCausalLM_input(Matrix3D<int> input_ids_, std::vector<Matrix3D<int8_t>> past_keys_,
                         std::vector<Matrix3D<int8_t>> past_values_)
        : input_ids(input_ids_), past_keys(past_keys_), past_values(past_values_) {
        has_past_keys_values = true;
    }
};

class OPTForCausalLM {
   public:
    OPTForCausalLM(std::string param_path, const struct model_config config);

    // std::string param_path, int voc_size_, int embed_dim_, int hidden_dim_, int num_heads_,
    //            int padding_idx_, int num_layers);
    // Int8OPTDecoder decoder, Matrix3D<float> lm_head): m_decoder(decoder), lm_head_weights(lm_head) {} // TODO: take a
    // decoder
    struct OPTForCausalLM_output forward(const struct OPTForCausalLM_input& input);

   private:
    Int8OPTDecoder decoder;
    Linear_FP lm_head;
    std::string profile_name = "OPTForCausalLM";
    float* logits_output;
    float* lm_head_weight;
};
