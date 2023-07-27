#include <iostream>

#include "OPTTokenizer.h"

void test_OPTEncode() {
    std::string bpe_file = "models/opt_merges.txt";
    std::string vocab_file = "models/opt_vocab.json";

    Encoder encoder = get_encoder(vocab_file, bpe_file);
    std::vector<int> encoded = encoder.encode(
        "Building a website can be done in 10 simple steps. This message is for general people, so we assume they "
        "don't have basic concepts.");
    std::vector<int> encoded_answer = {37500, 10,  998, 64, 28, 626, 11,   158, 2007, 2402, 4,  152,  1579,  16,
                                       13,    937, 82,  6,  98, 52,  6876, 51,  218,  75,   33, 3280, 14198, 4};
    bool is_equal = true;
    for (int i = 0; i < encoded.size(); i++) {
        if (encoded[i] != encoded_answer[i]) {
            is_equal = false;
            break;
        }
    }
    if (!is_equal)
        std::cout << "-------- Test of " << __func__ << ": Fail! -------- " << std::endl;
    else
        std::cout << "-------- Test of " << __func__ << ": Passed! -------- " << std::endl;
}

void test_OPTDecode() {
    std::string bpe_file = "models/opt_merges.txt";
    std::string vocab_file = "models/opt_vocab.json";
    ;

    Encoder encoder = get_encoder(vocab_file, bpe_file);
    std::vector<int> encoded_answer = {37500, 10,  998, 64, 28, 626, 11,   158, 2007, 2402, 4,  152,  1579,  16,
                                       13,    937, 82,  6,  98, 52,  6876, 51,  218,  75,   33, 3280, 14198, 4};
    std::string decoded = encoder.decode(encoded_answer);
    std::string decoded_answer =
        "Building a website can be done in 10 simple steps. This message is for general people, so we assume they "
        "don't have basic concepts.";
    bool is_equal = true;
    if (decoded != decoded_answer) is_equal = false;
    if (!is_equal)
        std::cout << "-------- Test of " << __func__ << ": Fail! -------- " << std::endl;
    else
        std::cout << "-------- Test of " << __func__ << ": Passed! -------- " << std::endl;
}

int main() {
    test_OPTEncode();
    test_OPTDecode();
};
