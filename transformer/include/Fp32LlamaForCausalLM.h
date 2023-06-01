#include "Fp32llamaDecoder.h"

struct Fp32LlamaForCausalLM_output {
    Matrix3D<float> logits;
    std::vector<Matrix3D<float>> past_keys, past_values;
};
struct Fp32LlamaForCausalLM_input {
    Matrix3D<int> input_ids;
    std::vector<Matrix3D<float>> past_keys, past_values;
    bool has_past_keys_values;

    Fp32LlamaForCausalLM_input(Matrix3D<int> input_ids_) : input_ids(input_ids_) { has_past_keys_values = false; }
    Fp32LlamaForCausalLM_input(Matrix3D<int> input_ids_, std::vector<Matrix3D<float>> past_keys_,
                               std::vector<Matrix3D<float>> past_values_)
        : input_ids(input_ids_), past_keys(past_keys_), past_values(past_values_) {
        has_past_keys_values = true;
    }
};

class Fp32LlamaForCausalLM {
   public:
    Fp32LlamaForCausalLM(std::string param_path, const struct model_config config);

    // std::string param_path, int voc_size_, int embed_dim_, int hidden_dim_, int num_heads_,
    //            int padding_idx_, int num_layers);
    // Fp32llamaDecoder decoder, Matrix3D<float> lm_head): m_decoder(decoder), lm_head_weights(lm_head) {} // TODO: take
    // a decoder
    struct Fp32LlamaForCausalLM_output forward(const struct Fp32LlamaForCausalLM_input& input);

   private:
    Fp32llamaDecoder decoder;
    Linear_FP lm_head;
    std::string profile_name = "Fp32LlamaForCausalLM";
    float* logits_output;
    float* lm_head_weight;
};
