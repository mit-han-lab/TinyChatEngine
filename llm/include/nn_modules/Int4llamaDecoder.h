#include <cstdlib>
#include <string>
#include <vector>

#include "Int4llamaDecoderLayer.h"
#include "common.h"
#include "operators.h"

struct Int4llamaDecoder_output {
#ifdef QM_CUDA
    Matrix3D<float16_t> last_hidden_state;
    std::vector<Matrix3D<float16_t>> past_keys, past_values;
#else
    Matrix3D<float> last_hidden_state;
    std::vector<Matrix3D<float>> past_keys, past_values;
#endif
};
struct Int4llamaDecoder_input {
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

    Int4llamaDecoder_input() {}
    Int4llamaDecoder_input(Matrix3D<int> input_ids_) : input_ids(input_ids_) { 
        has_past_keys_values = false; 
        is_llava = false;
    }
#ifdef QM_CUDA
    Int4llamaDecoder_input(Matrix3D<int> input_ids_, std::vector<Matrix3D<float16_t>> past_keys_,
                           std::vector<Matrix3D<float16_t>> past_values_)
#else
    Int4llamaDecoder_input(Matrix3D<int> input_ids_, std::vector<Matrix3D<float>> past_keys_,
                           std::vector<Matrix3D<float>> past_values_)
#endif
        : input_ids(input_ids_), past_keys(past_keys_), past_values(past_values_) {
        has_past_keys_values = true;
        is_llava = false;
    }
    Int4llamaDecoder_input(Matrix3D<int> input_ids_, Matrix3D<float> image_embed_, Matrix3D<int> second_input_ids_)
        : input_ids(input_ids_), image_embed(image_embed_), second_input_ids(second_input_ids_) {
        has_past_keys_values = false;
        is_llava = true;
    }
    Int4llamaDecoder_input(Matrix3D<int> input_ids_, Matrix3D<float> image_embed_)
        : input_ids(input_ids_), image_embed(image_embed_) {
        has_past_keys_values = false;
        is_llava = true;
    }
};

class Int4llamaDecoder {
   public:
    Int4llamaDecoder(std::string param_path, const struct model_config config);
    Int4llamaDecoder(){};
    Matrix3D<float> prepare_decoder_attention_mask(int length, int past_length);
    struct Int4llamaDecoder_output forward(std::string param_path, const struct Int4llamaDecoder_input& input);
    int voc_size, embed_dim, padding_idx, hidden_dim, num_heads;
    float rms_norm_eps;
    std::vector<Int4llamaDecoderLayer> layers;
    std::string profile_name = "Int4llamaDecoder";
#ifdef QM_CUDA
    void free_cuda_memory();
    Embedding embed_tokens;
    LlamaRMSNorm_cuda norm;

    float16_t* attention_mask_buf = nullptr;
    float16_t* last_hidden_states_buf = nullptr;
    float* hidden_states_buf = nullptr;
    float16_t* hidden_states_half_buf = nullptr;
#else
    Embedding embed_tokens;
    LlamaRMSNorm norm;

    float* attention_mask_buf;
    float* pos_embeds_buf;
    float* last_hidden_states_buf;
    float* hidden_states_buf;
    float* inputs_embeds_buf;
    float* first_input_ids_buf;
    float* image_embed_buf;
    float* second_input_ids_buf;
#endif
    float* norm_weight_ptr = nullptr;
};
