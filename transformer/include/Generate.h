#include <algorithm>
#include <cassert>
#include <cstdio>
#include <iostream>
#include <queue>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

#include "Fp32llamaForCausalLM.h"
#include "Int4llamaForCausalLM.h"
#include "OPTForCausalLM.h"
#include "OPTTokenizer.h"
#include "operators.h"
#include "utils.h"

inline std::mt19937 OPT_rng;

typedef struct OPT_token_data {
    int id;       // token id
    float logit;  // log-odds of the token
    float p;      // probability of the token
} OPT_token_data;

typedef struct OPT_token_data_array {
    OPT_token_data* data;
    size_t size;
    bool sorted;
} OPT_token_data_array;

struct opt_params {
    int32_t seed = -1;        // RNG seed
    int32_t n_threads = 1;    // TODO: fix this
    int32_t n_predict = 128;  // new tokens to predict
    int32_t n_parts = -1;     // amount of model parts (-1 = determine from model dimensions)
    int32_t n_ctx = 512;      // context size
    int32_t n_batch = 512;    // batch size for prompt processing (must be >=32 to use BLAS)
    int32_t n_keep = 0;       // number of tokens to keep from initial prompt
    int32_t n_vocab = 50272;  // vocabulary size

    // sampling parameters
    std::unordered_map<int, float> logit_bias;  // logit bias for specific tokens
    int32_t top_k = 40;                         // <= 0 to use vocab size
    float top_p = 0.95f;                        // 1.0 = disabled
    float tfs_z = 1.00f;                        // 1.0 = disabled
    float typical_p = 1.00f;                    // 1.0 = disabled
    float temp = 0.80f;                         // 1.0 = disabled
    float repeat_penalty = 1.10f;               // 1.0 = disabled
    int32_t repeat_last_n = 64;                 // last n tokens to penalize (0 = disable penalty, -1 = context size)
    float frequency_penalty = 0.00f;            // 0.0 = disabled
    float presence_penalty = 0.00f;             // 0.0 = disabled
    int mirostat = 0;                           // 0 = disabled, 1 = mirostat, 2 = mirostat 2.0
    float mirostat_tau = 5.00f;                 // target entropy
    float mirostat_eta = 0.10f;                 // learning rate
};

void sample_repetition_penalty(OPT_token_data_array* candidates, const int* last_tokens, size_t last_tokens_size,
                               float penalty);

void sample_frequency_and_presence_penalties(OPT_token_data_array* candidates, const int* last_tokens_p,
                                             size_t last_tokens_size, float alpha_frequency, float alpha_presence);

int sample_token_greedy(OPT_token_data_array* candidates);

void sample_temperature(OPT_token_data_array* candidates_p, float temp);

void sample_softmax(OPT_token_data_array* candidates);

int sample_token(OPT_token_data_array* candidates);

void sample_top_k(OPT_token_data_array* candidates, int k, size_t min_keep);

int sample_token_mirostat(const int n_vocab, OPT_token_data_array* candidates, float tau, float eta, int m, float* mu);

int sample_token_mirostat_v2(OPT_token_data_array* candidates, float tau, float eta, float* mu);

void sample_tail_free(OPT_token_data_array* candidates, float z, size_t min_keep);

void sample_typical(OPT_token_data_array* candidates, float p, size_t min_keep);

void sample_top_p(OPT_token_data_array* candidates, float p, size_t min_keep);

std::vector<int> OPTGenerate(OPTForCausalLM model, std::vector<int> input_ids,
                             const struct opt_params generation_config, Encoder* encoder = NULL,
                             bool interactive = false);

enum { OPT, LLaMA_FP32, LLaMA_INT4 };
std::vector<int> LLaMAGenerate(void* model, int model_type, std::string text, const struct opt_params generation_config,
                               std::string voc_path, bool interactive);
