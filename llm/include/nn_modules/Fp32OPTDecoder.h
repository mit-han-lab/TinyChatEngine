#include <cstdlib>
#include <string>
#include <vector>

#include "Fp32OPTDecoderLayer.h"
#include "common.h"
#include "operators.h"

struct Fp32OPTDecoder_output {
    Matrix3D<float> last_hidden_state;
    std::vector<Matrix3D<float>> past_keys, past_values;
};
struct Fp32OPTDecoder_input {
    Matrix3D<int> input_ids;
    std::vector<Matrix3D<float>> past_keys, past_values;
    bool has_past_keys_values;

    Fp32OPTDecoder_input(Matrix3D<int> input_ids_) : input_ids(input_ids_) { has_past_keys_values = false; }
    Fp32OPTDecoder_input(Matrix3D<int> input_ids_, std::vector<Matrix3D<float>> past_keys_,
                         std::vector<Matrix3D<float>> past_values_)
        : input_ids(input_ids_), past_keys(past_keys_), past_values(past_values_) {
        has_past_keys_values = true;
    }
};

class Fp32OPTDecoder {
   public:
    Fp32OPTDecoder(std::string param_path, const struct model_config config);
    Fp32OPTDecoder(){};
    Matrix3D<float> prepare_decoder_attention_mask(int length, int past_length);
    Matrix3D<float> get_position_embed(int sql_length, int past_length);
    struct Fp32OPTDecoder_output forward(const struct Fp32OPTDecoder_input& input);
    Embedding embed_tokens, embed_positions;
    int voc_size, embed_dim, padding_idx, hidden_dim, num_heads;
    std::vector<Fp32OPTDecoderLayer> layers;
    LayerNorm final_layer_norm;
    std::string profile_name = "Fp32OPTDecoder";

   private:
    float* attention_mask_buf;
    float* pos_embeds_buf;
    float* last_hidden_states_buf;
    float* hidden_states_buf;
};
