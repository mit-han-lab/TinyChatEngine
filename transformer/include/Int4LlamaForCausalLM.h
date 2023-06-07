#include "Int4llamaDecoder.h"

struct Int4LlamaForCausalLM_output {
    Matrix3D<float> logits;
    std::vector<Matrix3D<float>> past_keys, past_values;
};
struct Int4LlamaForCausalLM_input {
    Matrix3D<int> input_ids;
    std::vector<Matrix3D<float>> past_keys, past_values;
    bool has_past_keys_values;

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

    // std::string param_path, int voc_size_, int embed_dim_, int hidden_dim_, int num_heads_,
    //            int padding_idx_, int num_layers);
    // Int4llamaDecoder decoder, Matrix3D<float> lm_head): m_decoder(decoder), lm_head_weights(lm_head) {} // TODO: take
    // a decoder
    struct Int4LlamaForCausalLM_output forward(const struct Int4LlamaForCausalLM_input& input);

   private:
    Int4llamaDecoder decoder;
    Linear_FP_int4 lm_head;
    std::string profile_name = "Int4LlamaForCausalLM";
    float* logits_output;
    uint8_t* lm_head_weight;
};
