#include <iostream>
#include <map>
#include <httplib.h>

#include "Generate.h"

std::map<std::string, int> model_config = {
    {"OPT_125m", OPT_125M},       {"OPT_1.3B", OPT_1_3B}, {"OPT_6.7B", OPT_6_7B},         {"LLaMA_7B", LLaMA_7B},
    {"LLaMA2_7B_chat", LLaMA_7B}, {"7b", LLaMA_7B},       {"LLaMA2_13B_chat", LLaMA_13B}, {"13b", LLaMA_13B}};

std::map<std::string, std::string> model_path = {{"OPT_125m", "models/OPT_125m"},
                                                 {"OPT_1.3B", "models/OPT_1.3B"},
                                                 {"OPT_6.7B", "models/OPT_6.7B"},
                                                 {"LLaMA_7B", "models/LLaMA_7B"},
                                                 {"LLaMA2_7B_chat", "models/LLaMA_7B_2_chat"},
                                                 {"LLaMA2_13B_chat", "models/LLaMA_13B_2_chat"},
                                                 {"7b", "models/LLaMA_7B_2_chat"},
                                                 {"13b", "models/LLaMA_13B_2_chat"}};

std::map<std::string, int> data_format_list = {
    {"FP32", FP32}, {"INT8", QINT8}, {"INT4", INT4}, {"int4", INT4}, {"fp32", FP32},
};

bool isLLaMA(std::string s) {
    std::string LLaMA_prefix = "LLaMA";

    if (s.substr(0, LLaMA_prefix.size()) == LLaMA_prefix || s == "7b" || s == "13b")
        return true;
    else
        return false;
}
/// @brief  Om

/// @return 
int main() {
    httplib::Server svr;
    std::string target_model = "LLaMA2_7B_chat";
    std::string target_data_format = "INT4";
    Profiler::getInstance().for_demo = true;

    std::cout << "TinyChatEngine by MIT HAN Lab: https://github.com/mit-han-lab/TinyChatEngine" << std::endl;
    if (isLLaMA(target_model)) {
            std::cout << "Using model: " + target_model << std::endl;
            if (target_data_format == "INT4" || target_data_format == "int4")
                std::cout << "Using AWQ for 4bit quantization: https://github.com/mit-han-lab/llm-awq" << std::endl;
            else
                std::cout << "Using data format: " << target_data_format << std::endl;
    
      int format_id = data_format_list[target_data_format];

      // Load model
        std::cout << "Loading model... " << std::flush; 
        
        //std::cout -->standard output stream ,<<: This is the insertion operator, It's used to insert data into the output stream.
        //std::flush: This is a manipulator that flushes (clears) the output buffer associated with the stream. Normally, when you use std::cout to print text, the output is buffered, which means it's not immediately displayed on the console. Flushing the buffer ensures that the text is displayed immediately.
        
        int model_id = model_config[target_model];
        std::string m_path = model_path[target_model];
        #ifdef MODEL_PREFIX
        m_path = MODEL_PREFIX + m_path;
        #endif


    // Define an endpoint to handle incoming chat messages via HTTP POST
    svr.Post("/chat", [](const httplib::Request& req, httplib::Response& res) {
         std::string target_model = "LLaMA2_7B_chat";
         std::string target_data_format = "INT4";
        struct opt_params generation_config;
 
        // Load encoder
        std::string bpe_file = "models/opt_merges.txt";
        std::string vocab_file = "models/opt_vocab.json";
        Encoder encoder = get_encoder(vocab_file, bpe_file);
        std::string decode;
        generation_config.n_predict = 512;
        generation_config.n_vocab = 32000;
        generation_config.temp = 0.1f;
        generation_config.repeat_penalty = 1.25f;
        
        int model_id = model_config[target_model];
        std::string m_path = model_path[target_model];
            m_path = "INT4/" + m_path;

        // Extract chat message from the request
        std::string message = req.body;
        //std::getline(std::cin, message);
        std::cout << "input  message: "<<req.body<< std::endl;
           

        std::vector<int> input_ids = encoder.encode(req.body);
                    std::cout << "encode created: "<< std::endl;

        std::string decoded = encoder.decode(input_ids);
            std::cout << "decode created: "<< std::endl;

        Int4LlamaForCausalLM model = Int4LlamaForCausalLM(m_path, get_opt_model_config(model_id));
            std::cout << "Model created: "<< std::endl;

              //  LLaMAGenerate(m_path, &model, LLaMA_INT4, input, generation_config, "models/llama_vocab.bin", true, false);

        // Process the chat message (e.g., use your existing chat logic)
        // Modify this section to integrate with your chat generation code
        // Generate
        std::vector<int> generated_ids = OPTGenerate(&model, OPT_INT4, input_ids, generation_config, &encoder, true, false);
        decoded = encoder.decode(generated_ids);
        std::cout << "generated:" << decoded << std::endl;

        // Send a response (e.g., generated response or acknowledgment)
        std::string response =  decoded  ;// Modify this line

         


        res.set_content(response, "text/plain");
    });
    }
    std::cout << "Chat server is running. Listening on port 8080..." << std::endl;
    svr.listen("0.0.0.0", 8080); // Listen on all network interfaces

    return 0;
}
