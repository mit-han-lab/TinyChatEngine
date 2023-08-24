#ifndef OPT_TOKENIZER_H
#define OPT_TOKENIZER_H

#include <algorithm>
#include <cassert>
#include <cmath>
#include <codecvt>
#include <cstdio>
#include <fstream>
#include <locale>
#include <map>
#include <queue>
#include <random>
#include <regex>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
// #include <boost/regex.hpp> // Tricky to support this in windows
#include <nlohmann/json.hpp>

// std::vector<int> OPT_tokenize(const OPT_vocab & vocab, const std::string & text, bool add_bos);

struct pair_hash {
    template <class T1, class T2>
    std::size_t operator()(const std::pair<T1, T2> &p) const {
        auto h1 = std::hash<T1>{}(p.first);
        auto h2 = std::hash<T2>{}(p.second);
        return h1 ^ h2;
    }
};

class Encoder {
   public:
    Encoder(std::map<std::string, int> encoder, std::vector<std::pair<std::string, std::string>> bpe_merges);
    std::unordered_map<int, std::string> bytes_to_unicode();
    std::set<std::pair<std::string, std::string>> get_pairs(std::vector<std::string> word);
    std::string bpe(std::string token);
    std::vector<int> encode(std::string text);
    std::string decode(std::vector<int> tokens);

   private:
    std::map<std::string, int> encoder;
    std::map<int, std::string> decoder;
    std::unordered_map<int, std::string> byte_encoder;
    std::unordered_map<std::string, int> byte_decoder;
    std::unordered_map<std::pair<std::string, std::string>, int, pair_hash> bpe_ranks;
    std::unordered_map<std::string, std::string> cache;
};

Encoder get_encoder(std::string vocab_file, std::string bpe_file);

#endif
