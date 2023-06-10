#include <iostream>
#include <map>

#include "OPTGenerate.h"

std::map<std::string, int> model_config = {
    {"OPT125M", OPT_125M},
    {"OPT1.3B", OPT_1_3B},
    {"OPT6.7B", OPT_6_7B},
    {"LLaMA7B", LLaMA_7B},
};

std::map<int, std::string> model_path = {
    {OPT_125M, "models/OPT_125m"},
    {OPT_1_3B, "models/OPT_1.3B"},
    {OPT_6_7B, "models/OPT_6.7B"},
    {LLaMA_7B, "models/LLaMA_7B"},
};

std::map<std::string, int> data_format_list = {
    {"FP32", FP32},
    {"INT8", INT8},
    {"INT4", INT4},
};

int main(int argc, char* argv[]) {
    std::string target_model = "LLaMA7B";
    std::string target_data_format = "FP32";

    if (argc == 3) {
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

        auto data_format_input = argv[2];
        if (data_format_list.count(data_format_input) == 0) {
            std::cerr << "Data format:" << data_format_input << " unsupported" << std::endl;
            std::cerr << "Please select one of the following:";
            for (const auto& k : data_format_list) {
                std::cerr << k.first << ", ";
            }
            std::cerr << std::endl;
            throw("Unsupported data format\n");
        }
        std::cout << "Data format: " << argv[2] << " selected" << std::endl;
        target_data_format = argv[2];
    } else {
        if (target_model == "LLaMA7B") {
            std::cout << "Using model: " + target_model << std::endl;
            std::cout << "Using LLaMA's default data format: " + target_data_format << std::endl;
        } else {  // OPT
            target_model = "OPT6.7B";
            target_data_format = "INT8";
            std::cout << "Using model: " + target_model << std::endl;
            std::cout << "Using OPT's default data format: " + target_data_format << std::endl;
        }
    }

    if (target_model == "LLaMA7B") {
        int format_id = data_format_list[target_data_format];

        // Load model
        std::cout << "Loading model... " << std::flush;
        int model_id = model_config[target_model];
        std::string m_path = model_path[model_id];

        struct opt_params generation_config;
        generation_config.n_predict = 256;
        generation_config.n_vocab = 32000;

        if (format_id == FP32) {
            Fp32LlamaForCausalLM model = Fp32LlamaForCausalLM(m_path, get_opt_model_config(model_id));
            std::cout << "Finished!" << std::endl;

            // Get input from the user
            std::cout << "Please enter a line of text: ";
            std::string input;
            std::getline(std::cin, input);

            Fp32LLaMAGenerate(model, input, generation_config, "models/LLaMA_7B/ggml-vocab.bin", true);
        } 
        else if (format_id == INT4) {
            Int4LlamaForCausalLM model = Int4LlamaForCausalLM(m_path, get_opt_model_config(model_id));
            std::cout << "Finished!" << std::endl;

            // Get input from the user
            std::cout << "Please enter a line of text: ";
            std::string input;
            std::getline(std::cin, input);

            Int4LLaMAGenerate(model, input, generation_config, "models/LLaMA_7B/ggml-vocab.bin", true);
        }
        else {
            std::cout << std::endl;
            std::cerr << "At this time, we only support FP32 and INT4 for LLaMA7B." << std::endl;
        }
    } else {  // OPT
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
    }
};
