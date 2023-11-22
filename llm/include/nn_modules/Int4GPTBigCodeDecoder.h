#include <cstdlib>
#include <string>
#include <vector>

#include "Int4GPTBigCodeDecoderLayer.h"
#include "common.h"
#include "operators.h"

struct Int4GPTBigCodeDecoder_output {
    Matrix3D<float> last_hidden_state;
    std::vector<Matrix3D<float>> past_keys, past_values;
};
struct Int4GPTBigCodeDecoder_input {
    Matrix3D<int> input_ids;
    std::vector<Matrix3D<float>> past_keys, past_values;
    bool has_past_keys_values;

    Int4GPTBigCodeDecoder_input(Matrix3D<int> input_ids_) : input_ids(input_ids_) { has_past_keys_values = false; }
    Int4GPTBigCodeDecoder_input(Matrix3D<int> input_ids_, std::vector<Matrix3D<float>> past_keys_,
                         std::vector<Matrix3D<float>> past_values_)
        : input_ids(input_ids_), past_keys(past_keys_), past_values(past_values_) {
        has_past_keys_values = true;
    }
};

class Int4GPTBigCodeDecoder {
   public:
    Int4GPTBigCodeDecoder(std::string param_path, const struct model_config config);
    Int4GPTBigCodeDecoder(){};
    Matrix3D<float> prepare_decoder_attention_mask(int length, int past_length);
    Matrix3D<float> get_position_embed(int sql_length, int past_length);
    struct Int4GPTBigCodeDecoder_output forward(const struct Int4GPTBigCodeDecoder_input& input);
    Embedding wte, wpe;
    int voc_size, embed_dim, padding_idx, hidden_dim, num_heads, max_position_embeddings;
    std::vector<Int4GPTBigCodeDecoderLayer> layers;
    LayerNorm ln_f;
    std::string profile_name = "Int4GPTBigCodeDecoder";

   private:
    float* attention_mask_buf;
    float* pos_embeds_buf;
    float* last_hidden_states_buf;
    float* hidden_states_buf;
};
