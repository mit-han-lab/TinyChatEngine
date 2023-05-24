#include <iostream>
#include <map>

#include "OPTGenerate.h"

std::map<std::string, int> model_config = {
    {"OPT125M", OPT_125M},
    {"OPT1.3B", OPT_1_3B},
    {"OPT6.7B", OPT_6_7B},
};

std::map<int, std::string> model_path = {
    {OPT_125M, "models/OPT_125m"},
    {OPT_1_3B, "models/OPT_1.3B"},
    {OPT_6_7B, "models/OPT_6.7B"},
};

int main(int argc, char* argv[]) {
    std::string target_model = "OPT1.3B";

    if (argc > 1) {
        auto target_str = argv[1];
        if (model_config.count(target_model) == 0) {
            std::cerr << "Model config:" << target_str << " unsupported" << std::endl;
            std::cerr << "Please select one of the following:";
            for (const auto& k : model_config) {
                std::cerr << k.first << ", ";
            }
            std::cerr << std::endl;
            throw("Unsupported model\n");
        }
        std::cout << "Model: " << argv[1] << " selected" << std::endl;
        target_model = argv[1];
    } else {
        std::cout << "Using default model: " + target_model << std::endl;
    }

    // Load model
    std::cout << "Loading model... " << std::flush;
    int model_id = model_config[target_model];
    std::string m_path = model_path[model_id];
    OPTForCausalLM model = OPTForCausalLM(m_path, get_opt_model_config(model_id));
    std::cout << "Finished!" << std::endl;

    // Load encoder
    std::string vocab_file = "./models/OPT_125m/vocab.json";
    std::string bpe_file = "./models/OPT_125m/merges.txt";
    Encoder encoder = get_encoder(vocab_file, bpe_file);

    // Get input from the user
    std::cout << "Please enter a line of text: ";
    std::string input;
    std::getline(std::cin, input);
    std::vector<int> input_ids = encoder.encode(input);
    std::string decoded = encoder.decode(input_ids);
    std::cout << "input:" << decoded << std::endl;

    struct opt_params generation_config;
    generation_config.n_predict = 256;
    std::vector<int> generated_ids = OPTGenerate(model, input_ids, generation_config, &encoder, true);

    decoded = encoder.decode(generated_ids);
};
