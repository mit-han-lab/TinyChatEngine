#include "Fp32OPTDecoder.h"
#include "operators.h"
#include "utils.h"
#include "utils_memalloc.h"

int NUM_THREAD = 8;

void test_Fp32OPTDecoder() {
    const int embed_dim = 2048, sqlen = 2, b = 1;
    MemoryAllocator mem_buf;

    Matrix3D<int> input_ids(mem_buf.get_intbuffer(sqlen), b, 1, sqlen);
    input_ids.load("assets/OPT/tests/attn/OPT_1.3B/Fp32_decoder_1st_input_ids.bin");
    struct Fp32OPTDecoder_input input_1st = {input_ids};

    Fp32OPTDecoder decoder = Fp32OPTDecoder("FP32/models/OPT_1.3B/decoder", get_opt_model_config(OPT_1_3B));

    struct Fp32OPTDecoder_output output_1st = decoder.forward(input_1st);

    // reasoning phase: 1st run
    Matrix3D<float> last_hidden_state1_GT(mem_buf.get_fpbuffer(b * sqlen * embed_dim), b, sqlen, embed_dim);
    last_hidden_state1_GT.load("assets/OPT/tests/attn/OPT_1.3B/Fp32_decoder_1st_last_hidden_state.bin");

    print_first_k_elelment("Fp32_decoder_1st_last_hidden_state", last_hidden_state1_GT.m_data, 20);
    print_first_k_elelment("output_1st", output_1st.last_hidden_state.m_data, 20);
    bool success = check_two_equal(output_1st.last_hidden_state.m_data, last_hidden_state1_GT.m_data,
                                   last_hidden_state1_GT.length(), 1e-5);

    if (!success)
        std::cout << "Test of " << __func__ << ": Fail!" << std::endl;
    else
        std::cout << "-------- Test of " << __func__ << ": Passed! -------- " << std::endl;
}

int main() { test_Fp32OPTDecoder(); }
