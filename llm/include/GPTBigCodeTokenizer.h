/*

Adapted from llama.cpp and starcoder.cpp:
https://github.com/ggerganov/llama.cpp
https://github.com/bigcode-project/starcoder.cpp

*/

#ifndef GPTBIGCODE_TOKENIZER_H
#define GPTBIGCODE_TOKENIZER_H

#include <cstdint>
#include <cstdio>
#include <iostream>
#include <map>
#include <queue>
#include <string>
#include <unordered_map>
#include <vector>
#include <random>
#include <thread>
#include <fstream>

//
// Vocab utils
//

std::string trim(const std::string & s);

std::string replace(
        const std::string & s,
        const std::string & from,
        const std::string & to);

struct starcoder_vocab {
    std::map<std::string, int32_t> token_to_id;
    std::map<int32_t, std::string> id_to_token;
    std::vector<std::string> special_tokens;

    void add_special_token(const std::string & token);
};

/*
 *  Tokenizer
 */
starcoder_vocab starcoder_init_vocab(const std::string & vocab_file);

const char* starcoder_id_to_token(starcoder_vocab& vocab, int id);

int starcoder_tokenize(const starcoder_vocab &vocab, const std::string &text, std::vector<int> &tokens, int n_max_tokens);

#endif
