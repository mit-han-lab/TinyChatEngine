#include "Int4llamaDecoder.h"

struct Int4LlamaForCausalLM_output {
    Matrix3D<float> logits;
    std::vector<Matrix3D<float>> past_keys, past_values;
};
struct Int4LlamaForCausalLM_input {
    Matrix3D<int> input_ids;
    std::vector<Matrix3D<float>> past_keys, past_values;
    bool has_past_keys_values;

    Int4LlamaForCausalLM_input() {}
    Int4LlamaForCausalLM_input(Matrix3D<int> input_ids_) : input_ids(input_ids_) { has_past_keys_values = false; }
    Int4LlamaForCausalLM_input(Matrix3D<int> input_ids_, std::vector<Matrix3D<float>> past_keys_,
                               std::vector<Matrix3D<float>> past_values_)
        : input_ids(input_ids_), past_keys(past_keys_), past_values(past_values_) {
        has_past_keys_values = true;
    }
};

class Int4LlamaForCausalLM {
   public:
    Int4LlamaForCausalLM(std::string param_path, const struct model_config config);
    struct Int4LlamaForCausalLM_output forward(const struct Int4LlamaForCausalLM_input& input);

   private:
    Int4llamaDecoder decoder;
    Linear_FP_int4 lm_head;
    std::string profile_name = "Int4LlamaForCausalLM";
    float* logits_output;
    uint8_t* lm_head_weight;
};
