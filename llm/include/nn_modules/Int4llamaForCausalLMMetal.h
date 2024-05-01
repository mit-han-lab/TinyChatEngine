#include "Int4llamaDecoder.h"

struct Int4LlamaForCausalLM_output {
    Matrix3D<float> logits;
#ifdef QM_METAL
    std::vector<Matrix3D<float16_t>> past_keys, past_values;
#else
    std::vector<Matrix3D<float>> past_keys, past_values;
#endif
};
struct Int4LlamaForCausalLM_input {
    Matrix3D<int> input_ids;
    bool has_past_keys_values;
#ifdef QM_METAL
    std::vector<Matrix3D<float16_t>> past_keys, past_values;
#else
    std::vector<Matrix3D<float>> past_keys, past_values;
#endif

    Int4LlamaForCausalLM_input() {}
    Int4LlamaForCausalLM_input(Matrix3D<int> input_ids_) : input_ids(input_ids_) { has_past_keys_values = false; }
#ifdef QM_METAL
    Int4LlamaForCausalLM_input(Matrix3D<int> input_ids_, std::vector<Matrix3D<float16_t>> past_keys_,
                               std::vector<Matrix3D<float16_t>> past_values_)
#else
    Int4LlamaForCausalLM_input(Matrix3D<int> input_ids_, std::vector<Matrix3D<float>> past_keys_,
                               std::vector<Matrix3D<float>> past_values_)
#endif
        : input_ids(input_ids_), past_keys(past_keys_), past_values(past_values_) {
        has_past_keys_values = true;
    }
};

class Int4LlamaForCausalLM {
   public:
    Int4LlamaForCausalLM(std::string param_path, const struct model_config config);
    Int4LlamaForCausalLM(){};
    struct Int4LlamaForCausalLM_output forward(std::string param_path, const struct Int4LlamaForCausalLM_input& input);
    float* logits_output = nullptr;
#ifdef QM_METAL
    void free_cuda_memory();
    int* lm_head_weight = nullptr;
    float16_t* logits_output_half = nullptr;
#else
    uint8_t* lm_head_weight;
#endif

   private:
    std::string profile_name = "Int4LlamaForCausalLM";
    Int4llamaDecoder decoder;
#ifdef QM_METAL
    Linear_half_int4 lm_head;
#else
    Linear_FP_int4 lm_head;
#endif
};


// 1. modified the code to be suitable for Metal
// 2. investigate the problem of waituntilcompleted (multiple encoders in order inside command buffer)
// 3. found more kernels needed