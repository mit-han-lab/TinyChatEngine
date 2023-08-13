#include "Int4llamaDecoder.h"

struct Int4LlamaForCausalLM_output {
    Matrix3D<float> logits;
#ifdef USE_CUDA
    std::vector<Matrix3D<float16_t>> past_keys, past_values;
#else
    std::vector<Matrix3D<float>> past_keys, past_values;
#endif
};
struct Int4LlamaForCausalLM_input {
    Matrix3D<int> input_ids;
    bool has_past_keys_values;
#ifdef USE_CUDA
    std::vector<Matrix3D<float16_t>> past_keys, past_values;
#else
    std::vector<Matrix3D<float>> past_keys, past_values;
#endif

    Int4LlamaForCausalLM_input(Matrix3D<int> input_ids_) : input_ids(input_ids_) { has_past_keys_values = false; }

#ifdef USE_CUDA
    Int4LlamaForCausalLM_input(Matrix3D<int> input_ids_, std::vector<Matrix3D<float16_t>> past_keys_, std::vector<Matrix3D<float16_t>> past_values_)
#else
    Int4LlamaForCausalLM_input(Matrix3D<int> input_ids_, std::vector<Matrix3D<float>> past_keys_, std::vector<Matrix3D<float>> past_values_)
#endif
        : input_ids(input_ids_), past_keys(past_keys_), past_values(past_values_) {
        has_past_keys_values = true;
    }
};

class Int4LlamaForCausalLM {
   public:
    Int4LlamaForCausalLM(std::string param_path, const struct model_config config);
    Int4LlamaForCausalLM() {};
    void free_cuda_memory();

    // std::string param_path, int voc_size_, int embed_dim_, int hidden_dim_, int num_heads_,
    //            int padding_idx_, int num_layers);
    // Int4llamaDecoder decoder, Matrix3D<float> lm_head): m_decoder(decoder), lm_head_weights(lm_head) {} // TODO: take
    // a decoder
    struct Int4LlamaForCausalLM_output forward(const struct Int4LlamaForCausalLM_input& input);
    float* logits_output = nullptr;
#ifdef USE_CUDA
    int* lm_head_weight = nullptr;
    float16_t* logits_output_half = nullptr;
    // float16_t* split_8_buffer;
#else
    uint8_t* lm_head_weight;
#endif

   private:
    std::string profile_name = "Int4LlamaForCausalLM";
    Int4llamaDecoder decoder;
#ifdef USE_CUDA
    Linear_half_int4 lm_head;
#else
    Linear_FP_int4 lm_head;
#endif
};
