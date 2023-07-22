#include "OPTGenerate.h"
#include "LLaMATokenizer.h"
#include "common.h"
#include "utils.h"

// void Int4LLaMAFreeMemory() {
//     // Int4LlamaForCausalLM
//     cudaFree(logits_output_half);
//     cudaFree(logits_output);
//     cudaFree(lm_head_weight);

//     // Int4llamaDecoder
//     cudaFree(attention_mask_buf);
//     cudaFree(last_hidden_states_buf);
//     cudaFree(hidden_states_buf);
//     cudaFree(hidden_states_half_buf);
//     cudaFree(norm_weight_ptr);

//     // Int4llamaDecoderLayer
//     cudaFree(hidden_states_half_arr);
//     cudaFree(final_layer_norm_arr);
//     cudaFree(gate_proj_arr);
//     cudaFree(up_proj_arr);
//     cudaFree(down_proj_arr);
//     cudaFree(hidden_states_arr);
//     cudaFree(input_layernorm_weight_ptr);
//     cudaFree(post_attention_layernorm_ptr);
//     cudaFree(gate_proj_weight);
//     cudaFree(down_proj_weight);
//     cudaFree(up_proj_weight);

//     // Int4llamaAttention
//     cudaFree(attn_weights_arr);
//     cudaFree(attn_output_half_arr);
//     cudaFree(query_states_unshape_arr);
//     cudaFree(attn_output_arr);
//     cudaFree(attn_output_transpose_arr);
//     cudaFree(key_states_unshape_arr);
//     cudaFree(key_states_arr);
//     cudaFree(value_states_unshape_arr);
//     cudaFree(value_states_arr);
//     cudaFree(query_states_arr);
//     cudaFree(value_states_transpose_arr);
//     cudaFree(key_states_arr_cache);
//     cudaFree(value_states_arr_cache);
//     cudaFree(cos_buf);
//     cudaFree(sin_buf);
//     cudaFree(q_weight);
//     cudaFree(k_weight);
//     cudaFree(v_weight);
//     cudaFree(o_weight);
//     free(cache_num);
// }

// Int4LlamaForCausalLM
std::vector<int> Int4LLaMAGenerate(Int4LlamaForCausalLM model, std::string text, const struct opt_params generation_config,
                               std::string voc_path, bool interactive) {
    std::vector<int> last_n_tokens(generation_config.n_ctx);
    std::fill(last_n_tokens.begin(), last_n_tokens.end(), 0);
    std::vector<int> embd;
    std::vector<int> generate_ids;

    const int max_token = 2048;
    std::vector<int> input_ids(max_token);
    llama_vocab vocab = llama_init_vocab(voc_path.c_str());
    const int n = llama_tokenize(vocab, text.c_str(), input_ids.data(), input_ids.size(), true);
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

    if (interactive) std::cout << "Generated: " << std::endl;

    bool has_past_kv = false;
    std::vector<Matrix3D<half>> past_keys, past_values;
    int n_remain = generation_config.n_predict;
    while (n_remain != 0) {
        STATS_START("Token generation");
        std::vector<float> logits(generation_config.n_vocab);

        int sqlen = 1;
        struct Int4LlamaForCausalLM_output model_output;
        if (has_past_kv) {
            Matrix3D<int> input_ids_mat(input_ids.data(), 1, 1, sqlen);
            struct Int4LlamaForCausalLM_input model_input = {input_ids_mat, past_keys, past_values};
            model_output = model.forward(model_input);
        } else {
            sqlen = input_ids.size();
            Matrix3D<int> input_ids_mat(input_ids.data(), 1, 1, sqlen);
            struct Int4LlamaForCausalLM_input model_input = {input_ids_mat};
            model_output = model.forward(model_input);
        }
        past_keys = model_output.past_keys;
        past_values = model_output.past_values;
        has_past_kv = true;
        // memcpy model_ouput.logits[-1] to logits
        memcpy(logits.data(), &model_output.logits.m_data[(sqlen - 1) * generation_config.n_vocab],
               generation_config.n_vocab * sizeof(float));

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
        OPT_sample_repetition_penalty(&candidates_p, last_n_tokens.data() + last_n_tokens.size() - last_n_repeat,
                                      last_n_repeat, repeat_penalty);
        OPT_sample_frequency_and_presence_penalties(&candidates_p,
                                                    last_n_tokens.data() + last_n_tokens.size() - last_n_repeat,
                                                    last_n_repeat, alpha_frequency, alpha_presence);

        int id = 0;
        if (temp <= 0) {
            id = OPT_sample_token_greedy(&candidates_p);
        } else {
            if (mirostat == 1) {
                static float mirostat_mu = 2.0f * mirostat_tau;
                const int mirostat_m = 100;
                OPT_sample_temperature(&candidates_p, temp);
                id = OPT_sample_token_mirostat(n_vocab, &candidates_p, mirostat_tau, mirostat_eta, mirostat_m,
                                               &mirostat_mu);
            } else if (mirostat == 2) {
                static float mirostat_mu = 2.0f * mirostat_tau;
                OPT_sample_temperature(&candidates_p, temp);
                id = OPT_sample_token_mirostat_v2(&candidates_p, mirostat_tau, mirostat_eta, &mirostat_mu);
            } else {
                // Temperature sampling
                OPT_sample_top_k(&candidates_p, top_k, 1);
                OPT_sample_tail_free(&candidates_p, tfs_z, 1);
                OPT_sample_typical(&candidates_p, typical_p, 1);
                OPT_sample_top_p(&candidates_p, top_p, 1);
                OPT_sample_temperature(&candidates_p, temp);
                id = OPT_sample_token(&candidates_p);
            }
        }

        last_n_tokens.erase(last_n_tokens.begin());
        last_n_tokens.push_back(id);
        embd.push_back(id);
        generate_ids.push_back(id);
        input_ids = std::vector<int>{id};

        if (interactive) std::cout << llama_id_to_token(vocab, id) << std::flush;

        --n_remain;
        STATS_END("Token generation");
    }

    if (interactive) std::cout << std::endl;

    Profiler::getInstance().report_internal();
    Profiler::getInstance().reset();

    // Free memory
    // Int4LLaMAFreeMemory();

    return generate_ids;
}
