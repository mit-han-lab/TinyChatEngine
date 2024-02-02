#include "Int4llamaDecoder.h"

struct Int4LlamaForCausalLM_output {
    Matrix3D<float> logits;
#ifdef QM_CUDA
    std::vector<Matrix3D<float16_t>> past_keys, past_values;
#else
    std::vector<Matrix3D<float>> past_keys, past_values;
#endif
};
struct Int4LlamaForCausalLM_input {
    Matrix3D<int> input_ids;
    Matrix3D<float> image_embed;
    Matrix3D<int> second_input_ids;
    bool has_past_keys_values;
    bool is_llava;
#ifdef QM_CUDA
    std::vector<Matrix3D<float16_t>> past_keys, past_values;
#else
    std::vector<Matrix3D<float>> past_keys, past_values;
#endif

    Int4LlamaForCausalLM_input() {}
    Int4LlamaForCausalLM_input(Matrix3D<int> input_ids_) : input_ids(input_ids_) { 
        has_past_keys_values = false; 
        is_llava = false;
    }
#ifdef QM_CUDA
    Int4LlamaForCausalLM_input(Matrix3D<int> input_ids_, std::vector<Matrix3D<float16_t>> past_keys_,
                               std::vector<Matrix3D<float16_t>> past_values_)
#else
    Int4LlamaForCausalLM_input(Matrix3D<int> input_ids_, std::vector<Matrix3D<float>> past_keys_,
                               std::vector<Matrix3D<float>> past_values_)
#endif
        : input_ids(input_ids_), past_keys(past_keys_), past_values(past_values_) {
        has_past_keys_values = true;
        is_llava = false;
    }
    Int4LlamaForCausalLM_input(Matrix3D<int> input_ids_, Matrix3D<float> image_embed_, Matrix3D<int> second_input_ids_)
        : input_ids(input_ids_), image_embed(image_embed_), second_input_ids(second_input_ids_) {
        has_past_keys_values = false;
        is_llava = true;
    }
    Int4LlamaForCausalLM_input(Matrix3D<int> input_ids_, Matrix3D<float> image_embed_)
        : input_ids(input_ids_), image_embed(image_embed_) {
        has_past_keys_values = false;
        is_llava = true;
    }
};

class Int4LlamaForCausalLM {
   public:
    Int4LlamaForCausalLM(std::string param_path, const struct model_config config);
    Int4LlamaForCausalLM(){};
    struct Int4LlamaForCausalLM_output forward(std::string param_path, const struct Int4LlamaForCausalLM_input& input);
    float* logits_output = nullptr;
#ifdef QM_CUDA
    void free_cuda_memory();
    int* lm_head_weight = nullptr;
    float16_t* logits_output_half = nullptr;
#else
    uint8_t* lm_head_weight;
#endif

   private:
    std::string profile_name = "Int4LlamaForCausalLM";
    Int4llamaDecoder decoder;
#ifdef QM_CUDA
    Linear_half_int4 lm_head;
#else
    Linear_FP_int4 lm_head;
#endif
};
