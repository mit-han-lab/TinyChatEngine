#include <thread>
#include <string>
#include <sstream>
#include <stdio.h>

#include "Generate.h"
#include "LLaMATokenizer.h"
#include "common.h"
#include "utils.h"
#include "interface.h"

// Function to speak in the background
static void sayInBackground(const std::string& text) {
    std::string command = "./application/sts_utils/speak \"" + text + "\"";
    int result = std::system(command.c_str());
    (void)result;
}

typedef struct {
    char *str;
    int id;
} TokenIndex;

typedef struct {
    char** vocab;
    float* vocab_scores;
    TokenIndex *sorted_vocab;
    int vocab_size;
    unsigned int max_token_length;
    unsigned char byte_pieces[512]; // stores all single-byte strings
} Tokenizer;

int compare_tokens(const void *a, const void *b);
void build_tokenizer(Tokenizer* t, const char* tokenizer_path, int vocab_size);
void free_tokenizer(Tokenizer* t);
char* decode(Tokenizer* t, int token);
int str_lookup(char *str, TokenIndex *sorted_vocab, int vocab_size);
void encode(Tokenizer* t, const char *text, int8_t bos, int8_t eos, int *tokens, int *n_tokens);

std::string LLaMA3Generate(std::string param_path, void *model_ptr, int model_type, std::string text, const struct opt_params generation_config,
                          std::string voc_path, bool interactive, bool voicechat) {
    std::vector<int> last_n_tokens(generation_config.n_ctx);
    std::fill(last_n_tokens.begin(), last_n_tokens.end(), 0);
    std::vector<int> embd;
    std::vector<int> generate_ids;

    // llama_vocab vocab = llama_init_vocab(voc_path.c_str());
    // build the Tokenizer via the tokenizer .bin file
    Tokenizer tokenizer;
    build_tokenizer(&tokenizer, voc_path.c_str(), generation_config.n_vocab);

    const int max_token = 2048;
    std::vector<int> input_ids(max_token);
    // const int n = llama_tokenize(vocab, text.c_str(), input_ids.data(), input_ids.size(), true);
    int num_tokens = 0;
    // encode the user prompt into tokens
    encode(&tokenizer, text.c_str(), 0, 0, input_ids.data(), &num_tokens);
    input_ids.resize(num_tokens);

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

        if (id == 128001) {
            break_cnt--;
            continue;
        }  // eos
        else if (id == 128000)
            continue;
        break_cnt = 2;

        bool skip = false;
        if (id == 35075 or id == 128009) {  // token = "Human" or "|eot_id|"
            break_cnt = 0;
            skip = true;
        }

        last_n_tokens.erase(last_n_tokens.begin());
        last_n_tokens.push_back(id);
        embd.push_back(id);
        generate_ids.push_back(id);
        input_ids = std::vector<int>{id};

        if (interactive && !skip) {
            // output += llama_id_to_token(vocab, id);
            // std::cout << llama_id_to_token(vocab, id) << std::flush;
            output += decode(&tokenizer, id);
            std::cout << decode(&tokenizer, id) << std::flush;

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

    // Set prompt color
    set_print_yellow();
    Profiler::getInstance().report_internal();
    Profiler::getInstance().reset();
    // Reset color
    set_print_reset();

    // Free the tokenizer
    free_tokenizer(&tokenizer);
    
    return output;
}


// Adapted from llama3.c: https://github.com/jameswdelancey/llama3.c
// ----------------------------------------------------------------------------
// The Byte Pair Encoding (BPE) Tokenizer that translates strings <-> tokens
int compare_tokens(const void *a, const void *b) {
    return strcmp(((TokenIndex*)a)->str, ((TokenIndex*)b)->str);
}

void build_tokenizer(Tokenizer* t, const char* tokenizer_path, int vocab_size) {
    // i should have written the vocab_size into the tokenizer file... sigh
    t->vocab_size = vocab_size;
    // malloc space to hold the scores and the strings
    t->vocab = (char**)malloc(vocab_size * sizeof(char*));
    t->vocab_scores = (float*)malloc(vocab_size * sizeof(float));
    t->sorted_vocab = NULL; // initialized lazily
    for (int i = 0; i < 256; i++) {
        t->byte_pieces[i * 2] = (unsigned char)i;
        t->byte_pieces[i * 2 + 1] = '\0';
    }
    // read in the file
    FILE *file = fopen(tokenizer_path, "rb");
    if (!file) { fprintf(stderr, "couldn't load %s\n", tokenizer_path); exit(EXIT_FAILURE); }
    if (fread(&t->max_token_length, sizeof(int), 1, file) != 1) { fprintf(stderr, "failed read\n"); exit(EXIT_FAILURE); }
    int len;
    for (int i = 0; i < vocab_size; i++) {
        if (fread(t->vocab_scores + i, sizeof(float), 1, file) != 1) { fprintf(stderr, "failed read\n"); exit(EXIT_FAILURE);}
        if (fread(&len, sizeof(int), 1, file) != 1) { fprintf(stderr, "failed read\n"); exit(EXIT_FAILURE); }
        t->vocab[i] = (char *)malloc(len + 1);
        if (fread(t->vocab[i], len, 1, file) != 1) { fprintf(stderr, "failed read\n"); exit(EXIT_FAILURE); }
        t->vocab[i][len] = '\0'; // add the string terminating token
    }
    fclose(file);
}

void free_tokenizer(Tokenizer* t) {
    for (int i = 0; i < t->vocab_size; i++) { free(t->vocab[i]); }
    free(t->vocab);
    free(t->vocab_scores);
    free(t->sorted_vocab);
}

char* decode(Tokenizer* t, int token) {
    char *piece = t->vocab[token];

    // careful, some tokens designate raw bytes, and look like e.g. '<0x01>'
    // parse this and convert and return the actual byte
    unsigned char byte_val;
    if (sscanf(piece, "<0x%02hhX>", &byte_val) == 1) {
        piece = (char*)t->byte_pieces + byte_val * 2;
    }
    return piece;
}

int str_lookup(char *str, TokenIndex *sorted_vocab, int vocab_size) {
    // efficiently find the perfect match for str in vocab, return its index or -1 if not found
    TokenIndex tok = { .str = str }; // acts as the key to search for
    TokenIndex *res = (TokenIndex *) bsearch(&tok, sorted_vocab, vocab_size, sizeof(TokenIndex), compare_tokens);
    return res != NULL ? res->id : -1;
}

void encode(Tokenizer* t, const char *text, int8_t bos, int8_t eos, int *tokens, int *n_tokens) {
    // encode the string text (input) into an upper-bound preallocated tokens[] array
    // bos != 0 means prepend the BOS token (=1), eos != 0 means append the EOS token (=2)
    if (text == NULL) { fprintf(stderr, "cannot encode NULL text\n"); exit(EXIT_FAILURE); }

    if (t->sorted_vocab == NULL) {
        // lazily malloc and sort the vocabulary
        // t->sorted_vocab = malloc(t->vocab_size * sizeof(TokenIndex));
        t->sorted_vocab = new TokenIndex[t->vocab_size];
        for (int i = 0; i < t->vocab_size; i++) {
            t->sorted_vocab[i].str = t->vocab[i];
            t->sorted_vocab[i].id = i;
        }
        qsort(t->sorted_vocab, t->vocab_size, sizeof(TokenIndex), compare_tokens);
    }

    // create a temporary buffer that will store merge candidates of always two consecutive tokens
    // *2 for concat, +1 for null terminator +2 for UTF8 (in case max_token_length is 1)
    // char* str_buffer = malloc((t->max_token_length*2 +1 +2) * sizeof(char));
    char* str_buffer = new char[(t->max_token_length*2 +1 +2)];
    size_t str_len = 0;

    // start at 0 tokens
    *n_tokens = 0;

    // add optional BOS (=128000) token, if desired
    if (bos) tokens[(*n_tokens)++] = 128000;

    // add_dummy_prefix is true by default
    // so prepend a dummy prefix token to the input string, but only if text != ""
    // TODO: pretty sure this isn't correct in the general case but I don't have the
    // energy to read more of the sentencepiece code to figure out what it's doing





    // Okay UTF-8 time. This will get messy. Here is the reference from Wikipedia:
    // Code point ↔ UTF-8 conversion
    // First code point	Last code point	Byte 1	Byte 2	Byte 3	Byte 4
    // U+0000	U+007F	    0xxxxxxx
    // U+0080	U+07FF	    110xxxxx	10xxxxxx
    // U+0800	U+FFFF	    1110xxxx	10xxxxxx	10xxxxxx
    // U+10000	U+10FFFF    11110xxx	10xxxxxx	10xxxxxx	10xxxxxx

    // process the raw (UTF-8) byte sequence of the input string
    for (const char *c = text; *c != '\0'; c++) {

        // reset buffer if the current byte is ASCII or a leading byte
        // 0xC0 is 11000000, so (*c & 0xC0) keeps the first 2 bits and zeros the rest
        // 0x80 is 10000000
        // in UTF-8, all continuation bytes start with "10" in first two bits
        // so in English this is: "if this byte is not a continuation byte"
        if ((*c & 0xC0) != 0x80) {
            // this byte must be either a leading byte (11...) or an ASCII char (0x...)
            // => reset our location, as we're starting a new UTF-8 codepoint
            str_len = 0;
        }

        // append the current byte to the buffer
        str_buffer[str_len++] = *c; // ++ is post-increment, incremented after this line
        str_buffer[str_len] = '\0';

        // while the next character is a continuation byte, continue appending
        // but if there are too many of them, just stop to avoid overruning str_buffer size.
        if ((*(c+1) & 0xC0) == 0x80 && str_len < 4) {
            continue;
        }

        // ok c+1 is not a continuation byte, so we've read in a full codepoint
        int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);

        if (id != -1) {
            // we found this codepoint in vocab, add it as a token
            tokens[(*n_tokens)++] = id;
        } else {
            // byte_fallback encoding: just encode each byte as a token
            // +3 is here because the first 3 vocab elements are <unk>, <s>, </s>
            // so the individual bytes only start at index 3
            for (int i=0; i < str_len; i++) {
                tokens[(*n_tokens)++] = (unsigned char)str_buffer[i] + 3;
            }
        }
        str_len = 0; // protect against a sequence of stray UTF8 continuation bytes
    }

    // merge the best consecutive pair or triple each iteration, according to the scores in vocab_scores
    while (1) {
        float best_score = -1e10;
        int best_id = -1;
        int best_idx = -1;
        int best_len = 2; // length of the best merge sequence (2 for pair, 3 for triple)

        // first, try to find the best pair to merge
        for (int i = 0; i < (*n_tokens - 1); i++) {
            // check if we can merge the pair (tokens[i], tokens[i+1])
            snprintf(str_buffer, t->max_token_length*2 +1 +2 +1, "%s%s", t->vocab[tokens[i]], t->vocab[tokens[i+1]]);
            int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);
            if (id != -1 && t->vocab_scores[id] > best_score) {
                // this merge pair exists in vocab! record its score and position
                best_score = t->vocab_scores[id];
                best_id = id;
                best_idx = i;
            }
        }

        // if no pair was found, try to find the best triple to merge
        if (best_idx == -1) {
            for (int i = 0; i < (*n_tokens - 2); i++) {
                // check if we can merge the triple (tokens[i], tokens[i+1], tokens[i+2])
                snprintf(str_buffer, t->max_token_length*2 +1 +2 +1, "%s%s%s", t->vocab[tokens[i]], t->vocab[tokens[i+1]], t->vocab[tokens[i+2]]);
                int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);
                if (id != -1 && t->vocab_scores[id] > best_score) {
                    // this merge triple exists in vocab! record its score and position
                    best_score = t->vocab_scores[id];
                    best_id = id;
                    best_idx = i;
                    best_len = 3;
                }
            }
        }

        if (best_idx == -1) {
            break; // we couldn't find any more pairs or triples to merge, so we're done
        }

        // merge the consecutive pair or triple (best_idx, best_idx+1[, best_idx+2]) into new token best_id
        tokens[best_idx] = best_id;
        // delete token(s) at position best_idx+1 (and optionally best_idx+2), shift the entire sequence back
        for (int i = best_idx + 1; i < (*n_tokens - best_len + 1); i++) {
            tokens[i] = tokens[i + best_len - 1];
        }
        (*n_tokens) -= (best_len - 1); // token length decreased by the number of merged tokens minus one
    }

    // add optional EOS (=128001) token, if desired
    if (eos) tokens[(*n_tokens)++] = 128001;

    free(str_buffer);
}
