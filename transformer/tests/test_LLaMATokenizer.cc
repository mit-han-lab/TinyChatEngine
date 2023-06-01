#include "LLaMATokenizer.h"

static const std::map<std::string, std::vector<int>> &test_LLaMATokenizer() {
    static std::map<std::string, std::vector<int>> llama_answer = {
        /* 1. */ {
            "Hello World",
            {
                1,
                10994,
                2787,
            },
        },
        /* 2. */
        {
            " Hello World!",
            {
                1,
                15043,
                2787,
                29991,
            },
        },
        /* 3. */
        {
            "This is Tiny LLM Engine.",
            {
                1,
                4013,
                338,
                323,
                4901,
                365,
                26369,
                10863,
                29889,
            },
        },
        /* 4. */
        {
            "Please introduce Massachusetts Institute of Technology (MIT)",
            {
                1,
                12148,
                14944,
                16167,
                8907,
                310,
                17968,
                313,
                26349,
                29897,
            },
        },
        /* 5. */
        {
            "Building a website can be done in 10 simple steps. This message is for general people, so we assume "
            "they don't have basic concepts.",
            {
                1,   8893, 292,  263,  4700,  508, 367, 2309, 297, 29871, 29896, 29900, 2560, 6576, 29889, 910,   2643,
                338, 363,  2498, 2305, 29892, 577, 591, 5251, 896, 1016,  29915, 29873, 505,  6996, 22001, 29889,
            },
        },
    };

    return llama_answer;
};

int main(int argc, char **argv) {
    // load the vocab
    // TODO: Change the dependency on the vocab file in the future.
    // TODO: Remove ggml-vocab.bin, use tokenizer.model in LLaMA repo on Huggingface instead.
    const std::string fname = "./models/LLAMA_7B/ggml-vocab.bin";
    // fprintf(stderr, "Reading vocab from: '%s'\n", fname.c_str());
    llama_vocab vocab = llama_init_vocab(fname.c_str());

    bool is_equal;
    int test_count = 1;
    for (const auto &llama_answer : test_LLaMATokenizer()) {
        std::vector<int> input_ids(llama_answer.first.size());
        const int n = llama_tokenize(vocab, llama_answer.first.c_str(), input_ids.data(), input_ids.size(), true);
        input_ids.resize(n);

        is_equal = input_ids.size() == llama_answer.second.size();

        for (int i = 0; i < (int)input_ids.size() && is_equal; ++i) {
            if (input_ids[i] != llama_answer.second[i]) {
                is_equal = false;
            }
        }

        // Print token ids
        // fprintf(stderr, "Test %d:\n", test_count);
        // fprintf(stderr, "Expected token ids: ");
        // for (const auto & t : llama_answer.second) {
        //     fprintf(stderr, "%6d, ", t);
        // }
        // fprintf(stderr, "\n");
        // fprintf(stderr, "Got token ids:      ");
        // for (const auto & t : input_ids) {
        //     fprintf(stderr, "%6d, ", t);
        // }
        // fprintf(stderr, "\n");

        // // Print token strings
        // fprintf(stderr, "Expected token strings:  ");
        // for (const auto & t : llama_answer.first) {
        //     fprintf(stderr, "%c", t);
        // }
        // fprintf(stderr, "\n");
        // fprintf(stderr, "Got token strings:       ");
        // for (const auto & t : input_ids) {
        //     fprintf(stderr, "%s", llama_id_to_token(vocab, t));
        // }
        // fprintf(stderr, "\n");

        test_count++;
    }
    // fprintf(stderr, "\n");

    if (!is_equal)
        std::cout << "-------- Test of " << __func__ << ": Fail! -------- " << std::endl;
    else
        std::cout << "-------- Test of " << __func__ << ": Passed! -------- " << std::endl;
    // fprintf(stderr, "\n");

    return 0;
}
