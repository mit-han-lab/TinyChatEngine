#include "OPTTokenizer.h"

/*std::vector<int> OPT_tokenize(const OPT_vocab & vocab, const std::string & text, bool add_bos) {
    std::vector<int> res(text.size() + (int) add_bos);
    return res;
}*/

/*
 *  Tokenizer
 */
Encoder::Encoder(std::map<std::string, int> encoder, std::vector<std::pair<std::string, std::string>> bpe_merges) {
    this->encoder = encoder;
    for (auto &it : encoder) {
        this->decoder[it.second] = it.first;
    }
    this->byte_encoder = bytes_to_unicode();
    for (auto &it : byte_encoder) {
        this->byte_decoder[it.second] = it.first;
    }
    for (int i = 0; i < bpe_merges.size(); ++i) {
        this->bpe_ranks.insert(std::make_pair(bpe_merges[i], i));
    }
}

std::unordered_map<int, std::string> Encoder::bytes_to_unicode() {
    std::unordered_map<int, std::string> byte_to_unicode;

    // Range from '!' to '~'
    for (int b = '!'; b <= '~'; ++b) {
        byte_to_unicode[b] = std::string(1, static_cast<char>(b));
    }

    // ASCII control characters
    byte_to_unicode[0x00] = u8"\u0000";  // Null char
    byte_to_unicode[0x01] = u8"\u0001";  // Start of Heading
    byte_to_unicode[0x02] = u8"\u0002";  // Start of Text
    byte_to_unicode[0x03] = u8"\u0003";  // End of Text
    byte_to_unicode[0x04] = u8"\u0004";  // End of Transmission
    byte_to_unicode[0x05] = u8"\u0005";  // Enquiry
    byte_to_unicode[0x06] = u8"\u0006";  // Acknowledge
    byte_to_unicode[0x07] = u8"\u0007";  // Bell
    byte_to_unicode[0x08] = u8"\u0008";  // Back Space
    byte_to_unicode[0x09] = u8"\u0009";  // Horizontal Tab
    byte_to_unicode[0x0A] = u8"\u000A";  // Line Feed
    byte_to_unicode[0x0B] = u8"\u000B";  // Vertical Tab
    byte_to_unicode[0x0C] = u8"\u000C";  // Form Feed
    byte_to_unicode[0x0D] = u8"\u000D";  // Carriage Return
    byte_to_unicode[0x0E] = u8"\u000E";  // Shift Out / X-On
    byte_to_unicode[0x0F] = u8"\u000F";  // Shift In / X-Off
    byte_to_unicode[0x10] = u8"\u0010";  // Data Line Escape
    byte_to_unicode[0x11] = u8"\u0011";  // Device Control 1 (oft. XON)
    byte_to_unicode[0x12] = u8"\u0012";  // Device Control 2
    byte_to_unicode[0x13] = u8"\u0013";  // Device Control 3 (oft. XOFF)
    byte_to_unicode[0x14] = u8"\u0014";  // Device Control 4
    byte_to_unicode[0x15] = u8"\u0015";  // Negative Acknowledgement
    byte_to_unicode[0x16] = u8"\u0016";  // Synchronous Idle
    byte_to_unicode[0x17] = u8"\u0017";  // End of Transmit Block
    byte_to_unicode[0x18] = u8"\u0018";  // Cancel
    byte_to_unicode[0x19] = u8"\u0019";  // End of Medium
    byte_to_unicode[0x1A] = u8"\u001A";  // Substitute
    byte_to_unicode[0x1B] = u8"\u001B";  // Escape
    byte_to_unicode[0x1C] = u8"\u001C";  // File Separator
    byte_to_unicode[0x1D] = u8"\u001D";  // Group Separator
    byte_to_unicode[0x1E] = u8"\u001E";  // Record Separator
    byte_to_unicode[0x1F] = u8"\u001F";  // Unit Separator

    // Basic ASCII characters
    byte_to_unicode[0x20] = "Ġ";  // Space

    // Extended ASCII range
    byte_to_unicode[0xA0] = u8"\u00A0";  // Non-breaking space
    byte_to_unicode[0xA1] = u8"\u0104";  // Ą
    byte_to_unicode[0xA2] = u8"\u0138";  // ĸ
    byte_to_unicode[0xA3] = u8"\u0156";  // Ŗ
    byte_to_unicode[0xA4] = u8"\u00A4";  // Currency symbol
    byte_to_unicode[0xA5] = u8"\u0128";  // Ĩ

    return byte_to_unicode;
}

std::set<std::pair<std::string, std::string>> Encoder::get_pairs(std::vector<std::string> word) {
    std::set<std::pair<std::string, std::string>> pairs;
    std::string prev_string = word[0];
    for (int i = 1; i < word.size(); ++i) {
        pairs.insert(std::make_pair(prev_string, word[i]));
        prev_string = word[i];
    }

    return pairs;
}

std::string Encoder::bpe(std::string token) {
    auto it = this->cache.find(token);  // Find the token in the cache
    if (it != this->cache.end()) {      // If the token is in the cache
        return it->second;
    }

    std::vector<std::string> word;  // word = tuple(token)
    for (char c : token) {
        word.push_back(std::string(1, c));
    }

    std::set<std::pair<std::string, std::string>> pairs = get_pairs(word);

    if (pairs.empty()) return token;

    while (true) {
        std::pair<std::string, std::string> bigram;
        int min_index = std::numeric_limits<int>::max();  // Start with the highest possible int value

        for (const auto &pair : pairs) {
            auto it = this->bpe_ranks.find(pair);  // Find the pair in the map
            if (it != this->bpe_ranks.end()) {     // If the pair is in the map
                if (it->second < min_index) {      // If the current pair's value is less than the min_index
                    min_index = it->second;
                    bigram = pair;
                }
            }
        }

        if (min_index == std::numeric_limits<int>::max())  // No pair was found in bpe_ranks
            break;

        std::string first = bigram.first;
        std::string second = bigram.second;
        std::vector<std::string> new_word;
        int i = 0;
        while (i < word.size()) {
            auto it = std::find(word.begin() + i, word.end(), first);
            if (it == word.end()) {
                new_word.insert(new_word.end(), word.begin() + i, word.end());
                break;
            }
            int j = it - word.begin();
            new_word.insert(new_word.end(), word.begin() + i, word.begin() + j);
            i = j;

            if (word[i] == first && i < word.size() - 1 && word[i + 1] == second) {
                new_word.push_back(first + second);
                i += 2;
            } else {
                new_word.push_back(word[i]);
                i += 1;
            }
        }

        word = new_word;

        if (word.size() == 1)
            break;
        else
            pairs = get_pairs(word);
    }

    this->cache.insert(std::make_pair(token, std::accumulate(word.begin(), word.end(), std::string())));

    return token;
}

std::vector<int> Encoder::encode(std::string text) {
    std::vector<int> bpe_tokens;

    // Using Regex to tokenize
    // MACOS does not support p{L}\\p{N}, we may need different regex lib
    // std::regex pat = std::regex("'s|'t|'re|'ve|'m|'ll|'d| ?[a-zA-Z]+| ?[0-9]+|
    // ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+");
    std::regex pat = std::regex("'s|'t|'re|'ve|'m|'ll|'d| ?[a-zA-Z]+| ?[0-9]+| ?[^\\s]+|\\s+(?!\\S)|\\s+");

    std::sregex_iterator iter(text.begin(), text.end(), pat);
    std::sregex_iterator end;

    while (iter != end) {
        std::string token = iter->str();
        std::string encoded_token;

        for (char b : token) {
            for (auto &it : this->byte_encoder) {
                if (it.first == int(static_cast<int>(b))) {
                    encoded_token += it.second;
                    break;
                }
            }
        }

        bpe_tokens.push_back(encoder[encoded_token]);
        ++iter;
    }

    return bpe_tokens;
}

std::string Encoder::decode(std::vector<int> tokens) {
    std::string text;
    for (const auto &token : tokens) {
        if (this->decoder.find(token) != this->decoder.end()) {
            // TODO: currently using brutal method. Need to optimize.
            int i_flag = 1;
            if (int(this->decoder[token][0]) < '!' || int(this->decoder[token][0]) > '~') {
                text += " ";
                i_flag = 2;
            } else {
                text += std::string(1, this->decoder[token][0]);
            }

            for (int i = i_flag; i < this->decoder[token].size(); ++i) {
                text += std::string(1, this->decoder[token][i]);
            }
        }
    }

    return text;
}

Encoder get_encoder(std::string vocab_file, std::string bpe_file) {
    std::ifstream i(vocab_file);
    nlohmann::json j;
    i >> j;

    std::map<std::string, int> encoder = j.get<std::map<std::string, int>>();

    std::ifstream infile(bpe_file);
    std::string line;
    std::vector<std::pair<std::string, std::string>> bpe_merges;
    // Read and ignore the first line.
    if (!std::getline(infile, line)) {
        // TODO: Handle the error (for example, file might be empty).
    }
    while (std::getline(infile, line)) {
        std::istringstream iss(line);
        std::string a, b;
        if (!(iss >> a >> b)) {
            break;
        }  // error
        bpe_merges.push_back({a, b});
    }

    return Encoder(encoder, bpe_merges);
}
