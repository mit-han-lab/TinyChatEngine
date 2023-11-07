
#include "Generate.h"
#include "GPTBigCodeTokenizer.h"
#include "common.h"
#include "utils.h"
#include <thread>
#include <string>
#include <sstream>

std::string GPTBigCodeGenerate(std::string param_path, void *model_ptr, int model_type, std::string text, const struct opt_params generation_config,
                          std::string voc_path, bool interactive, bool voicechat) {
    std::vector<int> last_n_tokens(generation_config.n_ctx);
    std::fill(last_n_tokens.begin(), last_n_tokens.end(), 0);
    std::vector<int> embd;
    std::vector<int> generate_ids;

    const int max_token = 2048;
    std::vector<int> input_ids(max_token);
    starcoder_vocab vocab = starcoder_init_vocab(voc_path);
    const int n = starcoder_tokenize(vocab, text, input_ids, input_ids.size());
    // printf("n = %d\n", n);
    input_ids.resize(n);

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
        if (model_type == StarCoder_INT4) {
            Int4GPTBigCodeForCausalLM *model = static_cast<Int4GPTBigCodeForCausalLM *>(model_ptr);
            struct Int4GPTBigCodeForCausalLM_output model_output;
            struct Int4GPTBigCodeForCausalLM_input model_input;
            if (has_past_kv) {
                Matrix3D<int> input_ids_mat(input_ids.data(), 1, 1, sqlen);
                model_input = {input_ids_mat, past_keys, past_values};
            } else {
                Matrix3D<int> input_ids_mat(input_ids.data(), 1, 1, sqlen);
                model_input = {input_ids_mat};
            }
            if (!new_prompt) STATS_START("Inference latency");
            // printf("Before model->forward\n");
            model_output = model->forward(param_path, model_input);
            // printf("After model->forward\n");
            if (!new_prompt) STATS_END("Inference latency");
            past_keys = model_output.past_keys;
            past_values = model_output.past_values;
            // memcpy model_ouput.logits[-1] to logits
            // printf("Before memcpy\n");
            memcpy(logits.data(), &model_output.logits.m_data[(sqlen - 1) * generation_config.n_vocab],
                   generation_config.n_vocab * sizeof(float));
            // printf("After memcpy\n");
        } else if (model_type == StarCoder_FP32) {
            Fp32GPTBigCodeForCausalLM *model = static_cast<Fp32GPTBigCodeForCausalLM *>(model_ptr);
            struct Fp32GPTBigCodeForCausalLM_output model_output;
            struct Fp32GPTBigCodeForCausalLM_input model_input;
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

        last_n_tokens.erase(last_n_tokens.begin());
        last_n_tokens.push_back(id);
        embd.push_back(id);
        generate_ids.push_back(id);
        input_ids = std::vector<int>{id};

        if (interactive && !skip) {
            output += starcoder_id_to_token(vocab, id);
            std::cout << starcoder_id_to_token(vocab, id) << std::flush;
        }

        new_prompt = false;
        --n_remain;
    }

    if (interactive) std::cout << std::endl;

    if (!voicechat) Profiler::getInstance().report_internal();
    Profiler::getInstance().reset();
    return output;
}
