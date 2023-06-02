#include "LLaMATokenizer.h"

/*
 *  Interface implementation
 */
struct llama_file {
    FILE * fp;
    size_t size;

    llama_file(const char * fname, const char * mode) {
        fp = std::fopen(fname, mode);
        if (!fp){
            std::cout << "opening " << fname << " fails" << std::endl;
            perror("Error opening file");
        }
    }

    void read_raw(void * ptr, size_t size) {
        if (size == 0) {
            return;
        }
        std::size_t ret = std::fread(ptr, size, 1, fp);
    }

    std::uint32_t read_u32() {
        std::uint32_t ret;
        read_raw(&ret, sizeof(ret));
        return ret;
    }

    std::string read_string(std::uint32_t len) {
        std::vector<char> chars(len);
        read_raw(chars.data(), len);
        return std::string(chars.data(), len);
    }
};

// TODO: Change the dependency on the vocab file in the future. 
// TODO: Remove ggml-vocab.bin, use tokenizer.model in LLaMA repo on Huggingface instead.
llama_vocab llama_init_vocab(const char * vocab_file) {
    llama_vocab vocab;
    int n_vocab = 32000;
    llama_file file(vocab_file, "rb");
    
    vocab.id_to_token.resize(n_vocab);

    // Read dummy data
    for (int i = 0; i < 9; i++) {
        file.read_u32();
    }

    for (uint32_t i = 0; i < n_vocab; i++) {
        uint32_t len = file.read_u32();
        std::string word = file.read_string(len);

        float score = 0.0f;
        file.read_raw(&score, sizeof(score));

        vocab.token_to_id[word] = i;

        auto & tok_score = vocab.id_to_token[i];
        tok_score.tok = std::move(word);
        tok_score.score = score;
    }

    return vocab;
}

/*
 *  Tokenizer
 */
const char * llama_id_to_token(const llama_vocab & vocab, int id) {
    if (id >= 32000) {
        return nullptr;
    }

    return vocab.id_to_token[id].tok.c_str();
}

struct llama_tokenizer {
    llama_tokenizer(const llama_vocab & vocab): vocab_(vocab) {}

    void tokenize(const std::string & text, std::vector<int32_t> & output) {
        // split string into utf8 chars
        int index = 0;
        size_t offs = 0;
        while (offs < text.size()) {
            llama_sp_symbol sym;
            size_t char_len = std::min(text.size() - offs, utf8_len(text[offs]));
            sym.text = text.c_str() + offs;
            sym.n = char_len;
            offs += char_len;
            sym.prev = index - 1;
            sym.next = offs == text.size() ? -1 : index + 1;
            index++;
            symbols_.emplace_back(sym);
        }

        // seed the work queue with all possible 2-character tokens.
        for (size_t i = 1; i < symbols_.size(); ++i) {
            try_add_bigram(i - 1, i);
        }

        // keep substituting the highest frequency pairs for as long as we can.
        while (!work_queue_.empty()) {
            auto bigram = work_queue_.top();
            work_queue_.pop();

            auto & left_sym = symbols_[bigram.left];
            auto & right_sym = symbols_[bigram.right];

            // if one of the symbols already got merged, skip it.
            if (left_sym.n == 0 || right_sym.n == 0 ||
                left_sym.n + right_sym.n != bigram.size) {
                continue;
            }

            // merge the right sym into the left one
            left_sym.n += right_sym.n;
            right_sym.n = 0;

            //printf("left = '%*s' size = %zu\n", (int) left_sym.n, left_sym.text, bigram.size);

            // remove the right sym from the chain
            left_sym.next = right_sym.next;
            if (right_sym.next >= 0) {
                symbols_[right_sym.next].prev = bigram.left;
            }

            // find more substitutions
            try_add_bigram(left_sym.prev, bigram.left);
            try_add_bigram(bigram.left, left_sym.next);
        }

        for (int i = 0; i != -1; i = symbols_[i].next) {
            auto & symbol = symbols_[i];
            auto token = vocab_.token_to_id.find(std::string(symbol.text, symbol.n));

            if (token == vocab_.token_to_id.end()) {
                // output any symbols that did not form tokens as bytes.
                for (int j = 0; j < (int) symbol.n; ++j) {
                    int32_t token_id = static_cast<uint8_t>(symbol.text[j]) + 3;
                    output.push_back(token_id);
                }
            } else {
                output.push_back((*token).second);
            }
        }
    }

private:
    void try_add_bigram(int left, int right) {
        if (left == -1 || right == -1) {
            return;
        }

        const std::string text = std::string(symbols_[left].text, symbols_[left].n + symbols_[right].n);
        auto token = vocab_.token_to_id.find(text);

        if (token == vocab_.token_to_id.end()) {
            return;
        }

        if (static_cast<size_t>((*token).second) >= vocab_.id_to_token.size()) {
            return;
        }

        const auto &tok_score = vocab_.id_to_token[(*token).second];

        llama_sp_bigram bigram;
        bigram.left = left;
        bigram.right = right;
        bigram.score = tok_score.score;
        bigram.size = text.size();
        work_queue_.push(bigram);
    }

    const llama_vocab & vocab_;
    std::vector<llama_sp_symbol> symbols_;
    llama_sp_bigram::queue work_queue_;
};

static std::vector<int32_t> llama_tokenize(const llama_vocab & vocab, const std::string & text, bool bos) {
    llama_tokenizer tokenizer(vocab);
    std::vector<int32_t> output;

    if (text.empty()) {
        return output;
    }

    if (bos) {
        output.push_back(llama_token_bos());
    }

    tokenizer.tokenize(text, output);
    return output;
}

int llama_tokenize(const llama_vocab & vocab,
                          const char * text,
                                 int * tokens,
                                 int   n_max_tokens,
                                bool   add_bos) {
    auto res = llama_tokenize(vocab, text, add_bos);

    if (n_max_tokens < (int) res.size()) {
        fprintf(stderr, "%s: too many tokens\n", __func__);
        return -((int) res.size());
    }

    for (size_t i = 0; i < res.size(); i++) {
        tokens[i] = res[i];
    }

    return res.size();
}
