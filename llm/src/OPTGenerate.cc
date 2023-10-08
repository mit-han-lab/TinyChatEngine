#include "Generate.h"
#include "common.h"
#include "utils.h"
#include <thread>
#include <string>
#include <sstream>

// Function to speak in the background
void speakInBackground(const std::string& text) {
    std::string command = "./application/sts_utils/speak \"" + text + "\"";
    int result = std::system(command.c_str());
    (void)result;
}

// OPTGenerate function
std::vector<int> OPTGenerate(void *model_ptr, int model_type, std::vector<int> input_ids,
                             const struct opt_params generation_config, Encoder *encoder, bool interactive, bool voicechat) {
    std::vector<int> last_n_tokens(generation_config.n_ctx);
    std::fill(last_n_tokens.begin(), last_n_tokens.end(), 0);
    std::vector<int> embd;
    std::vector<int> generate_ids;

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

    if (encoder == NULL) interactive = false;
    if (interactive) std::cout << "ASSISTANT: " << std::endl;

    bool has_past_kv = false;
    std::vector<Matrix3D<int8_t>> past_keys_int8, past_values_int8;
    std::vector<Matrix3D<float>> past_keys, past_values;
    int n_remain = generation_config.n_predict;
    std::string output;
    while (n_remain != 0) {
        STATS_START("Token generation");
        std::vector<float> logits(generation_config.n_vocab);

        int sqlen = 1;
        if (model_type == OPT_INT8) {
            struct OPTForCausalLM_output model_output;
            OPTForCausalLM *model = static_cast<OPTForCausalLM *>(model_ptr);
            if (has_past_kv) {
                Matrix3D<int> input_ids_mat(input_ids.data(), 1, 1, sqlen);
                struct OPTForCausalLM_input model_input = {input_ids_mat, past_keys_int8, past_values_int8};
                model_output = model->forward(model_input);
            } else {
                sqlen = input_ids.size();
                Matrix3D<int> input_ids_mat(input_ids.data(), 1, 1, sqlen);
                struct OPTForCausalLM_input model_input = {input_ids_mat};
                model_output = model->forward(model_input);
            }
            past_keys_int8 = model_output.past_keys;
            past_values_int8 = model_output.past_values;
            has_past_kv = true;
            // memcpy model_ouput.logits[-1] to logits
            memcpy(logits.data(), &model_output.logits.m_data[(sqlen - 1) * generation_config.n_vocab],
                   generation_config.n_vocab * sizeof(float));
        } else if (model_type == OPT_FP32) {
            struct Fp32OPTForCausalLM_output model_output;
            Fp32OPTForCausalLM *model = static_cast<Fp32OPTForCausalLM *>(model_ptr);
            if (has_past_kv) {
                Matrix3D<int> input_ids_mat(input_ids.data(), 1, 1, sqlen);
                struct Fp32OPTForCausalLM_input model_input = {input_ids_mat, past_keys, past_values};
                model_output = model->forward(model_input);
            } else {
                sqlen = input_ids.size();
                Matrix3D<int> input_ids_mat(input_ids.data(), 1, 1, sqlen);
                struct Fp32OPTForCausalLM_input model_input = {input_ids_mat};
                model_output = model->forward(model_input);
            }
            past_keys = model_output.past_keys;
            past_values = model_output.past_values;
            has_past_kv = true;
            // memcpy model_ouput.logits[-1] to logits
            memcpy(logits.data(), &model_output.logits.m_data[(sqlen - 1) * generation_config.n_vocab],
                   generation_config.n_vocab * sizeof(float));
        } else if (model_type == OPT_INT4) {
            struct Int4OPTForCausalLM_output model_output;
            Int4OPTForCausalLM *model = static_cast<Int4OPTForCausalLM *>(model_ptr);
            if (has_past_kv) {
                Matrix3D<int> input_ids_mat(input_ids.data(), 1, 1, sqlen);
                struct Int4OPTForCausalLM_input model_input = {input_ids_mat, past_keys, past_values};
                model_output = model->forward(model_input);
            } else {
                sqlen = input_ids.size();
                Matrix3D<int> input_ids_mat(input_ids.data(), 1, 1, sqlen);
                struct Int4OPTForCausalLM_input model_input = {input_ids_mat};
                model_output = model->forward(model_input);
            }
            past_keys = model_output.past_keys;
            past_values = model_output.past_values;
            has_past_kv = true;
            // memcpy model_ouput.logits[-1] to logits
            memcpy(logits.data(), &model_output.logits.m_data[(sqlen - 1) * generation_config.n_vocab],
                   generation_config.n_vocab * sizeof(float));
        }
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

        last_n_tokens.erase(last_n_tokens.begin());
        last_n_tokens.push_back(id);
        embd.push_back(id);
        generate_ids.push_back(id);
        input_ids = std::vector<int>{id};
        if (interactive) {
            output += encoder->decode(input_ids);
            std::cout << encoder->decode(input_ids) << std::flush;
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
                    std::thread sayThread(speakInBackground, output_copy);
                    sayThread.detach(); 
                } 
            } 
        }
        --n_remain;
        STATS_END("Token generation");
    }
    if (interactive && voicechat){
        speakInBackground(output);
    }
    if (interactive) std::cout << std::endl;

    if (!voicechat) Profiler::getInstance().report_internal();
    Profiler::getInstance().reset();

    return generate_ids;
}
