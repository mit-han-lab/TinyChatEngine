/*

Adapted from llama.cpp and starcoder.cpp:
https://github.com/ggerganov/llama.cpp
https://github.com/bigcode-project/starcoder.cpp

*/

#include <cmath>
#include <regex>

#include "GPTBigCodeTokenizer.h"

/*
 *  Interface implementation
 */
starcoder_vocab starcoder_init_vocab(const std::string & vocab_file) {
    starcoder_vocab vocab;
    int n_vocab = 49152;

    auto fin = std::ifstream(vocab_file, std::ios::binary);
    if (!fin) {
        fprintf(stderr, "%s: failed to open '%s'\n", __func__, vocab_file.c_str());
    }

    // Read dummy data
    for (int i = 0; i < 8; i++) {
        uint32_t dummy;
        fin.read((char *) &dummy, sizeof(dummy));
    }

    std::string word;
    std::vector<char> buf(128);
    for (uint32_t i = 0; i < n_vocab; i++) {
        uint32_t len;
        fin.read((char *) &len, sizeof(len));

        buf.resize(len);
        fin.read((char *) buf.data(), len);
        word.assign(buf.data(), len);

        vocab.token_to_id[word] = i;
        vocab.id_to_token[i] = word;
    }

    return vocab;
}

/*
 *  Tokenizer
 */
const char *starcoder_id_to_token(starcoder_vocab &vocab, int id) {
    if (id >= 49152) {
        return nullptr;
    }

    return vocab.id_to_token[id].c_str();
}

int starcoder_tokenize(const starcoder_vocab &vocab, const std::string &text, std::vector<int> &final_tokens, int n_max_tokens) {
    std::vector<std::string> words;
    std::vector<int32_t> tokens;

    // first split the text into words
    {
        std::string str = text;
        std::string pat = R"('s|'t|'re|'ve|'m|'ll|'d| ?[[:alpha:]]+| ?[[:digit:]]+| ?[^\s[:alpha:][:digit:]]+|\s+(?!\S)|\s+)";

        // Generate the subpattern from the special_tokens vector if it's not empty
        if (!vocab.special_tokens.empty()) {
            std::string special_tokens_subpattern;
            for (const auto & token : vocab.special_tokens) {
                if (!special_tokens_subpattern.empty()) {
                    special_tokens_subpattern += "|";
                }
                special_tokens_subpattern += token;
            }

            // Modify the regex pattern with the generated special tokens subpattern
            pat = special_tokens_subpattern + "|" + pat;
        }

        std::regex re(pat);
        std::smatch m;

        while (std::regex_search(str, m, re)) {
            for (auto x : m) {
                words.push_back(x);
            }
            str = m.suffix();
        }
    }

    // find the longest tokens that form the words:
    for (const auto & word : words) {
        if (word.size() == 0) continue;

        int i = 0;
        int n = word.size();
        while (i < n) {
            int j = n;
            while (j > i) {
                auto it = vocab.token_to_id.find(word.substr(i, j-i));
                if (it != vocab.token_to_id.end()) {
                    tokens.push_back(it->second);
                    i = j;
                    break;
                }
                --j;
            }
            if (i == n) {
                break;
            }
            if (j == i) {
                auto sub = word.substr(i, 1);
                if (vocab.token_to_id.find(sub) != vocab.token_to_id.end()) {
                    tokens.push_back(vocab.token_to_id.at(sub));
                } else {
                    fprintf(stderr, "%s: unknown token '%s'\n", __func__, sub.data());
                }
                ++i;
            }
        }
    }

    for (size_t i = 0; i < tokens.size(); i++) {
        final_tokens[i] = tokens[i];
    }

    return tokens.size();
}
