#include <iostream>

#include "Generate.h"

int main() {
    // std::vector<int> input_ids = {37500, 10,  998, 64, 28, 626, 11,   158, 2007, 2402, 4,  152,  1579,  16,
    //                                    13,    937, 82,  6,  98, 52,  6876, 51,  218,  75,   33, 3280, 14198, 4};
    std::string vocab_file = "./models/OPT_125m/vocab.json";
    std::string bpe_file = "./models/OPT_125m/merges.txt";

    Encoder encoder = get_encoder(vocab_file, bpe_file);
    std::vector<int> input_ids = encoder.encode("John went to MIT and study Computer Science.");

    std::string decoded = encoder.decode(input_ids);
    std::cout << "input:" << decoded << std::endl;

    OPTForCausalLM model = OPTForCausalLM("models/OPT_125m", get_opt_model_config(OPT_125M));
    const struct opt_params generation_config;
    std::vector<int> generated_ids = OPTGenerate(model, input_ids, generation_config);

    decoded = encoder.decode(generated_ids);
    std::cout << "generated:" << decoded << std::endl;
};
