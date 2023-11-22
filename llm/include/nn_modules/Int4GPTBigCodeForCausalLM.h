#include "Int4GPTBigCodeDecoder.h"

struct Int4GPTBigCodeForCausalLM_output {
    Matrix3D<float> logits;
    std::vector<Matrix3D<float>> past_keys, past_values;
};
struct Int4GPTBigCodeForCausalLM_input {
    Matrix3D<int> input_ids;
    std::vector<Matrix3D<float>> past_keys, past_values;
    bool has_past_keys_values;

    Int4GPTBigCodeForCausalLM_input() {}
    Int4GPTBigCodeForCausalLM_input(Matrix3D<int> input_ids_) : input_ids(input_ids_) { has_past_keys_values = false; }
    Int4GPTBigCodeForCausalLM_input(Matrix3D<int> input_ids_, std::vector<Matrix3D<float>> past_keys_,
                             std::vector<Matrix3D<float>> past_values_)
        : input_ids(input_ids_), past_keys(past_keys_), past_values(past_values_) {
        has_past_keys_values = true;
    }
};

class Int4GPTBigCodeForCausalLM {
   public:
    Int4GPTBigCodeForCausalLM(std::string param_path, const struct model_config config);
    struct Int4GPTBigCodeForCausalLM_output forward(std::string param_path, const struct Int4GPTBigCodeForCausalLM_input& input);

   private:
    Int4GPTBigCodeDecoder decoder;
    Linear_FP_int4 lm_head;
    std::string profile_name = "Int4GPTBigCodeForCausalLM";
    float* logits_output;
    uint8_t* lm_head_weight;
};
