#include "Int4llamaDecoder.h"

struct Int4LlamaForCausalLM_output {
    Matrix3D<float> logits;
    std::vector<Matrix3D<float16_t>> past_keys, past_values;
};
struct Int4LlamaForCausalLM_input {
    Matrix3D<int> input_ids;
    std::vector<Matrix3D<float16_t>> past_keys, past_values;
    bool has_past_keys_values;

    Int4LlamaForCausalLM_input(Matrix3D<int> input_ids_) : input_ids(input_ids_) { has_past_keys_values = false; }
    Int4LlamaForCausalLM_input(Matrix3D<int> input_ids_, std::vector<Matrix3D<float16_t>> past_keys_,
                               std::vector<Matrix3D<float16_t>> past_values_)
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
    Linear_half_int4 lm_head;
    //Linear_FP lm_head;
    std::string profile_name = "Int4LlamaForCausalLM";
    
    float16_t* logits_output_half;
    float* logits_output;
    // uint8_t* lm_head_weight;
    int* lm_head_weight;
    //float* lm_head_weight;
};
