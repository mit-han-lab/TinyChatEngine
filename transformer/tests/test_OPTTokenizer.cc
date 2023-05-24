#include <iostream>

#include "OPTTokenizer.h"

void test_OPTTokenizer() {
    // Test bpe
    // std::cout << "Test End!" << std::endl;
}

int main() {
    // test_OPTTokenizer();
    std::string vocab_file = "./models/OPT_125m/vocab.json";
    std::string bpe_file = "./models/OPT_125m/merges.txt";

    Encoder encoder = get_encoder(vocab_file, bpe_file);
    // encoder.new_bytes_to_unicode();
    // encoder.encode("Building. a don't.");
    // encoder.encode("Building a website can be done in 10 simple steps. This message is for general people, so we
    // assume they don't have basic concepts.");
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
    if (is_equal)
        std::cout << "-------- Test of Encoder::encode: Passed! -------- " << std::endl;
    else
        std::cout << "-------- Test of Encoder::encode: Failed! -------- " << std::endl;

    std::string decoded = encoder.decode(encoded);
    std::string decoded_answer =
        "Building a website can be done in 10 simple steps. This message is for general people, so we assume they "
        "don't have basic concepts.";
    is_equal = true;
    if (decoded != decoded_answer) is_equal = false;
    if (is_equal)
        std::cout << "-------- Test of Encoder::decode: Passed! -------- " << std::endl;
    else
        std::cout << "-------- Test of Encoder::decode: Failed! -------- " << std::endl;

    std::cout << "-------- End of test_OPTTokenizer --------" << std::endl;
};
