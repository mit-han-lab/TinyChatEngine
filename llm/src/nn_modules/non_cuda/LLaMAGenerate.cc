
#include "Generate.h"
#include "LLaMATokenizer.h"
#include "common.h"
#include "utils.h"
#include <thread>
#include <string>
#include <sstream>


// Function to speak in the background
void sayInBackground(const std::string& text) {
    std::string command = "./application/sts_utils/speak \"" + text + "\"";
    int result = std::system(command.c_str());
    (void)result;
}

std::string LLaMAGenerate(std::string param_path, void *model_ptr, int model_type, std::string text, const struct opt_params generation_config,
                          std::string voc_path, bool interactive, bool voicechat) {
    std::vector<int> last_n_tokens(generation_config.n_ctx);
    std::fill(last_n_tokens.begin(), last_n_tokens.end(), 0);
    std::vector<int> embd;
    std::vector<int> generate_ids;

    const int max_token = 2048;
    std::vector<int> input_ids(max_token);
    llama_vocab vocab = llama_init_vocab(voc_path.c_str());
    const int n = llama_tokenize(vocab, text.c_str(), input_ids.data(), input_ids.size(), true);
    input_ids.resize(n);

    bool is_codellama = false;
    if (param_path.find("CodeLLaMA") != std::string::npos) {
        is_codellama = true;
    }

    int n_consumed = 0;
    while ((int)input_ids.size() > n_consumed) {
        embd.push_back(input_ids[n_consumed]);
        last_n_tokens.erase(last_n_tokens.begin());
        last_n_tokens.push_back(input_ids[n_consumed]);
        ++n_consumed;

        if ((int)embd.size() >= generation_config.n_batch) {
            break;
        }
    }
    // if (interactive) std::cout << "ASSISTANT: " << std::endl;

    bool previous_two_hash = false;
    int break_cnt = 2;
    bool new_prompt = true;
    static bool has_past_kv = false;
    static std::vector<Matrix3D<float>> past_keys, past_values;
    int n_remain = generation_config.n_predict;
    std::string output;
    while (n_remain != 0 && break_cnt) {
        std::vector<float> logits(generation_config.n_vocab);

        int sqlen = 1;
        if (new_prompt) {
            sqlen = input_ids.size();
        }
        if (model_type == LLaMA_INT4) {
            Int4LlamaForCausalLM *model = static_cast<Int4LlamaForCausalLM *>(model_ptr);
            struct Int4LlamaForCausalLM_output model_output;
            struct Int4LlamaForCausalLM_input model_input;
            if (has_past_kv) {
                Matrix3D<int> input_ids_mat(input_ids.data(), 1, 1, sqlen);
                model_input = {input_ids_mat, past_keys, past_values};
            } else {
                Matrix3D<int> input_ids_mat(input_ids.data(), 1, 1, sqlen);
                model_input = {input_ids_mat};
            }
            if (!new_prompt) STATS_START("Inference latency");
            model_output = model->forward(param_path, model_input);
            if (!new_prompt) STATS_END("Inference latency");
            past_keys = model_output.past_keys;
            past_values = model_output.past_values;
            // memcpy model_ouput.logits[-1] to logits
            memcpy(logits.data(), &model_output.logits.m_data[(sqlen - 1) * generation_config.n_vocab],
                   generation_config.n_vocab * sizeof(float));
        } else if (model_type == LLaMA_FP32) {
            Fp32LlamaForCausalLM *model = static_cast<Fp32LlamaForCausalLM *>(model_ptr);
            struct Fp32LlamaForCausalLM_output model_output;
            struct Fp32LlamaForCausalLM_input model_input;
            if (has_past_kv) {
                Matrix3D<int> input_ids_mat(input_ids.data(), 1, 1, sqlen);
                model_input = {input_ids_mat, past_keys, past_values};
            } else {
                Matrix3D<int> input_ids_mat(input_ids.data(), 1, 1, sqlen);
                model_input = {input_ids_mat};
            }
            if (!new_prompt) STATS_START("Inference latency");
            model_output = model->forward(model_input);
            if (!new_prompt) STATS_END("Inference latency");
            past_keys = model_output.past_keys;
            past_values = model_output.past_values;
            // memcpy model_ouput.logits[-1] to logits
            memcpy(logits.data(), &model_output.logits.m_data[(sqlen - 1) * generation_config.n_vocab],
                   generation_config.n_vocab * sizeof(float));
        }
        has_past_kv = true;

        // Generate
        const int n_ctx = generation_config.n_ctx;
        const float temp = generation_config.temp;
        const int32_t top_k = generation_config.top_k <= 0 ? generation_config.n_vocab : generation_config.top_k;
        const float top_p = generation_config.top_p;
        const float tfs_z = generation_config.tfs_z;
        const float typical_p = generation_config.typical_p;
        const int32_t repeat_last_n = generation_config.repeat_last_n < 0 ? n_ctx : generation_config.repeat_last_n;
        const float repeat_penalty = generation_config.repeat_penalty;
        const float alpha_presence = generation_config.presence_penalty;
        const float alpha_frequency = generation_config.frequency_penalty;
        const int mirostat = generation_config.mirostat;
        const float mirostat_tau = generation_config.mirostat_tau;
        const float mirostat_eta = generation_config.mirostat_eta;
        const int n_vocab = generation_config.n_vocab;

        std::vector<OPT_token_data> candidates;
        candidates.reserve(n_vocab);
        for (int token_id = 0; token_id < n_vocab; token_id++) {
            candidates.emplace_back(OPT_token_data{token_id, logits[token_id], 0.0f});
        }

        OPT_token_data_array candidates_p = {candidates.data(), candidates.size(), false};

        // Apply penalties
        auto last_n_repeat = std::min(std::min((int)last_n_tokens.size(), repeat_last_n), n_ctx);
        sample_repetition_penalty(&candidates_p, last_n_tokens.data() + last_n_tokens.size() - last_n_repeat,
                                  last_n_repeat, repeat_penalty);
        sample_frequency_and_presence_penalties(&candidates_p,
                                                last_n_tokens.data() + last_n_tokens.size() - last_n_repeat,
                                                last_n_repeat, alpha_frequency, alpha_presence);

        int id = 0;
        if (temp <= 0) {
            id = sample_token_greedy(&candidates_p);
        } else {
            if (mirostat == 1) {
                static float mirostat_mu = 2.0f * mirostat_tau;
                const int mirostat_m = 100;
                sample_temperature(&candidates_p, temp);
                id =
                    sample_token_mirostat(n_vocab, &candidates_p, mirostat_tau, mirostat_eta, mirostat_m, &mirostat_mu);
            } else if (mirostat == 2) {
                static float mirostat_mu = 2.0f * mirostat_tau;
                sample_temperature(&candidates_p, temp);
                id = sample_token_mirostat_v2(&candidates_p, mirostat_tau, mirostat_eta, &mirostat_mu);
            } else {
                // Temperature sampling
                sample_top_k(&candidates_p, top_k, 1);
                sample_tail_free(&candidates_p, tfs_z, 1);
                sample_typical(&candidates_p, typical_p, 1);
                sample_top_p(&candidates_p, top_p, 1);
                sample_temperature(&candidates_p, temp);
                id = sample_token(&candidates_p);
            }
        }

        if (id == 2) {
            break_cnt--;
            continue;
        }  // eos
        else if (id == 1)
            continue;
        break_cnt = 2;

        bool skip = false;
        if (id == 2277 && !previous_two_hash) {
            previous_two_hash = true;
            skip = true;
        } else if (previous_two_hash && id == 29937) {  // token = #
            break_cnt = 0;
            skip = true;
        } else {
            if (previous_two_hash) std::cout << "##" << std::endl;
            previous_two_hash = false;
        }

        if (is_codellama && new_prompt) {
            new_prompt = false;
            // continue;
        }

        last_n_tokens.erase(last_n_tokens.begin());
        last_n_tokens.push_back(id);
        embd.push_back(id);
        generate_ids.push_back(id);
        input_ids = std::vector<int>{id};

        if (interactive && !skip) {
            output += llama_id_to_token(vocab, id);
            std::cout << llama_id_to_token(vocab, id) << std::flush;
            if (voicechat) {
                // Remove quotes
                output.erase(std::remove(output.begin(), output.end(), '\"'), output.end());
                // Remove hashtags
                output.erase(std::remove(output.begin(), output.end(), '#'), output.end());
                // Remove dashes
                std::replace(output.begin(), output.end(), '-', ' ');

                size_t lastPos;
                // starts ealier but slows down dictation
                bool ended = false;
                if (output.find(", ") != std::string::npos){
                    lastPos = output.rfind(',');
                    ended = true;
                }
                if (output.find("\n") != std::string::npos){
                    lastPos = output.rfind('\n');
                    ended = true;
                }
                else if (output.find(". ") != std::string::npos){
                    lastPos = output.rfind('.');
                    ended = true;
                }
                else if (output.find("! ") != std::string::npos){
                    lastPos = output.rfind('!');
                    ended = true;
                }
                else if (output.find("? ") != std::string::npos){
                    lastPos = output.rfind('?');
                    ended = true;
    
                }
                else if (output.find(": ") != std::string::npos){
                    lastPos = output.rfind(':');
                    ended = true;
                }
                if (ended){
                    // Extract sentence 1 (up to and including the last period)
                    std::string output_copy = output.substr(0, lastPos + 1);
                    // Extract beginning of sentence 2 (excluding the space after the last period)
                    output = output.substr(lastPos + 1); // Skip the last period and space
                    std::thread sayThread(sayInBackground, output_copy);
                    sayThread.detach(); 
                } 
            } 
        }

        new_prompt = false;
        --n_remain;
    }
    if (voicechat && interactive){
        sayInBackground(output);
    }

    if (interactive) std::cout << std::endl;

    Profiler::getInstance().report_internal();
    Profiler::getInstance().reset();
    return output;
}
