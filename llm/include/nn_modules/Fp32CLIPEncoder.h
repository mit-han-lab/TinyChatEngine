#include <cstdlib>
#include <string>
#include <vector>

#include "Fp32CLIPEncoderLayer.h"
#include "common.h"
#include "operators.h"

struct Fp32CLIPEncoder_output {
    Matrix3D<float> last_hidden_state;
    std::vector<Matrix3D<float>> past_keys, past_values;
};
struct Fp32CLIPEncoder_input {
    Matrix3D<float> hidden_states;
    Matrix3D<float> attention_mask;
    std::vector<Matrix3D<float>> past_keys, past_values;
    bool has_past_keys_values;

    Fp32CLIPEncoder_input(Matrix3D<float> hidden_states_, Matrix3D<float> attention_mask_)
        : hidden_states(hidden_states_), attention_mask(attention_mask_) { 
        has_past_keys_values = false; 
    }
    Fp32CLIPEncoder_input(Matrix3D<float> hidden_states_, Matrix3D<float> attention_mask_, 
                          std::vector<Matrix3D<float>> past_keys_, std::vector<Matrix3D<float>> past_values_)
        : hidden_states(hidden_states_), attention_mask(attention_mask_), past_keys(past_keys_), past_values(past_values_) {
        has_past_keys_values = true;
    }
};

class Fp32CLIPEncoder {
   public:
    Fp32CLIPEncoder(std::string param_path, const struct model_config config);
    Fp32CLIPEncoder(){};
    struct Fp32CLIPEncoder_output forward(const struct Fp32CLIPEncoder_input& input);
    std::vector<Fp32CLIPEncoderLayer> layers;
    std::string profile_name = "Fp32CLIPEncoder";
};
