#include <cstdlib>
#include <string>
#include <vector>

#include "Fp32CLIPEncoder.h"
#include "common.h"
#include "operators.h"

struct Fp32CLIPVisionTransformer_output {
    Matrix3D<float> last_hidden_state;
    std::vector<Matrix3D<float>> past_keys, past_values;
};
struct Fp32CLIPVisionTransformer_input {
    Matrix3D<float> input_image;
    std::vector<Matrix3D<float>> past_keys, past_values;
    bool has_past_keys_values;

    Fp32CLIPVisionTransformer_input() {}
    Fp32CLIPVisionTransformer_input(Matrix3D<float> input_image_) : input_image(input_image_) { has_past_keys_values = false; }
    Fp32CLIPVisionTransformer_input(Matrix3D<float> input_image_, std::vector<Matrix3D<float>> past_keys_,
                           std::vector<Matrix3D<float>> past_values_)
        : input_image(input_image_), past_keys(past_keys_), past_values(past_values_) {
        has_past_keys_values = true;
    }
};

class Fp32CLIPVisionTransformer {
   public:
    Fp32CLIPVisionTransformer(std::string param_path, const struct model_config config);
    Fp32CLIPVisionTransformer(){};
    struct Fp32CLIPVisionTransformer_output forward(const struct Fp32CLIPVisionTransformer_input& input);
    Embedding embed_positions;
    Conv2D embed_patch;
    LayerNorm pre_layernorm;
    Linear_FP mm_proj_0, mm_proj_2;
    int voc_size, embed_dim, padding_idx, hidden_dim, num_heads, image_size, patch_size, num_patches, num_positions, 
        projection_dim, mmproj_dim;
    std::vector<Fp32CLIPEncoderLayer> layers;
    std::string profile_name = "Fp32CLIPVisionTransformer";

   private:
    Fp32CLIPEncoder encoder;
    float* patch_embeds_buf;
    float* class_embeds_buf;
    float* pos_embeds_buf;
    float* last_hidden_states_buf;
    float* hidden_states_buf;
    float* embeddings_buf;
    float* mm_proj_0_arr;
    float* mm_proj_2_arr;
};
