#include <iostream>
#include <map>

#include "Generate.h"

std::map<std::string, int> model_config = {{"OPT_125m", OPT_125M},     {"OPT_1.3B", OPT_1_3B},
                                           {"OPT_6.7B", OPT_6_7B},     {"LLaMA_7B", LLaMA_7B},
                                           {"LLaMA_7B_AWQ", LLaMA_7B}, {"LLaMA_7B_2_chat", LLaMA_7B}};

std::map<std::string, std::string> model_path = {
    {"OPT_125m", "models/OPT_125m"},         {"OPT_1.3B", "models/OPT_1.3B"},
    {"OPT_6.7B", "models/OPT_6.7B"},         {"LLaMA_7B", "models/LLaMA_7B"},
    {"LLaMA_7B_AWQ", "models/LLaMA_7B_AWQ"}, {"LLaMA_7B_2_chat", "models/LLaMA_7B_2_chat"},
};

std::map<std::string, int> data_format_list = {
    {"FP32", FP32},
    {"INT8", INT8},
    {"INT4", INT4},
};

bool isLLaMA7B(std::string s) {
    std::string LLaMA_prefix = "LLaMA_7B";

    if (s.substr(0, LLaMA_prefix.size()) == LLaMA_prefix)
        return true;
    else
        return false;
}

int main(int argc, char* argv[]) {
    std::string target_model = "LLaMA_7B_2_chat";
    std::string target_data_format = "INT4";

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
        if (isLLaMA7B(target_model)) {
            std::cout << "Using model: " + target_model << std::endl;
            std::cout << "Using LLaMA's default data format: " + target_data_format << std::endl;
        } else {  // OPT
            target_model = "OPT6.7B";
            target_data_format = "INT8";
            std::cout << "Using model: " + target_model << std::endl;
            std::cout << "Using OPT's default data format: " + target_data_format << std::endl;
        }
    }

    if (isLLaMA7B(target_model)) {
        int format_id = data_format_list[target_data_format];

        // Load model
        std::cout << "Loading model... " << std::flush;
        int model_id = model_config[target_model];
        std::string m_path = model_path[target_model];

        struct opt_params generation_config;
        generation_config.n_predict = 512;
        generation_config.n_vocab = 32000;
        generation_config.temp = 0.1f;
        generation_config.repeat_penalty = 1.25f;

        if (format_id == FP32) {
            Fp32LlamaForCausalLM model = Fp32LlamaForCausalLM(m_path, get_opt_model_config(model_id));
            std::cout << "Finished!" << std::endl;

            // Get input from the user
            while (true) {
                std::cout << "USER: ";
                std::string input;
                std::getline(std::cin, input);
                input = "A chat between a human and an assistant.\n\n### Human: " + input + "\n### Assistant: \n";

                LLaMAGenerate(&model, LLaMA_FP32, input, generation_config, "models/llama_vocab.bin", true);
            }
        } else if (format_id == INT4) {
            m_path = "INT4/" + m_path;
            Int4LlamaForCausalLM model = Int4LlamaForCausalLM(m_path, get_opt_model_config(model_id));
            std::cout << "Finished!" << std::endl;

            // Get input from the user
            while (true) {
                std::cout << "USER: ";
                std::string input;
                std::getline(std::cin, input);
                input = "A chat between a human and an assistan.\n\n### Human: " + input + "\n### Assistant: \n";

                LLaMAGenerate(&model, LLaMA_INT4, input, generation_config, "models/llama_vocab.bin", true);
            }
        } else {
            std::cout << std::endl;
            std::cerr << "At this time, we only support FP32 and INT4 for LLaMA7B." << std::endl;
        }
    } else {  // OPT
        // Load model
        std::cout << "Loading model... " << std::flush;
        int model_id = model_config[target_model];
        std::string m_path = model_path[target_model];
        OPTForCausalLM model = OPTForCausalLM(m_path, get_opt_model_config(model_id));
        std::cout << "Finished!" << std::endl;

        // Load encoder
        std::string bpe_file = "models/opt_merges.txt";
        std::string vocab_file = "models/opt_vocab.json";
        Encoder encoder = get_encoder(vocab_file, bpe_file);

        // Get input from the user
        std::cout << "USER: ";
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
