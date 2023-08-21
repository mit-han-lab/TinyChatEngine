#include "Fp32OPTDecoder.h"

struct Fp32OPTForCausalLM_output {
    Matrix3D<float> logits;
    std::vector<Matrix3D<float>> past_keys, past_values;
};
struct Fp32OPTForCausalLM_input {
    Matrix3D<int> input_ids;
    std::vector<Matrix3D<float>> past_keys, past_values;
    bool has_past_keys_values;

    Fp32OPTForCausalLM_input(Matrix3D<int> input_ids_) : input_ids(input_ids_) { has_past_keys_values = false; }
    Fp32OPTForCausalLM_input(Matrix3D<int> input_ids_, std::vector<Matrix3D<float>> past_keys_,
                             std::vector<Matrix3D<float>> past_values_)
        : input_ids(input_ids_), past_keys(past_keys_), past_values(past_values_) {
        has_past_keys_values = true;
    }
};

class Fp32OPTForCausalLM {
   public:
    Fp32OPTForCausalLM(std::string param_path, const struct model_config config);
    struct Fp32OPTForCausalLM_output forward(const struct Fp32OPTForCausalLM_input& input);

   private:
    Fp32OPTDecoder decoder;
    Linear_FP lm_head;
    std::string profile_name = "Fp32OPTForCausalLM";
    float* logits_output;
    float* lm_head_weight;
};
