#ifndef LLaMA_TOKENIZER_H
#define LLaMA_TOKENIZER_H

#include <cstdint>
#include <cstdio>
#include <iostream>
#include <map>
#include <queue>
#include <string>
#include <unordered_map>
#include <vector>

static int llama_token_bos() { return 1; }

static int llama_token_eos() { return 2; }

static int llama_token_nl() { return 13; }

struct llama_vocab {
    struct token_score {
        std::string tok;
        float score;
    };

    std::unordered_map<std::string, int32_t> token_to_id;
    std::vector<token_score> id_to_token;
};

/*
 *  Tokenizer
 */
static size_t utf8_len(char src) {
    const size_t lookup[] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 4};
    uint8_t highbits = static_cast<uint8_t>(src) >> 4;

    return lookup[highbits];
}

struct llama_sp_symbol {
    using index = int;
    index prev;
    index next;
    const char* text;
    size_t n;
};

struct llama_sp_bigram {
    struct comparator {
        bool operator()(llama_sp_bigram& l, llama_sp_bigram& r) {
            return (l.score < r.score) || (l.score == r.score && l.left > r.left);
        }
    };
    using queue_storage = std::vector<llama_sp_bigram>;
    using queue = std::priority_queue<llama_sp_bigram, queue_storage, comparator>;
    llama_sp_symbol::index left;
    llama_sp_symbol::index right;
    float score;
    size_t size;
};

llama_vocab llama_init_vocab(const char* vocab_file);

const char* llama_id_to_token(const llama_vocab& vocab, int id);

int llama_tokenize(const llama_vocab& vocab, const char* text, int* tokens, int n_max_tokens, bool add_bos);

#endif
