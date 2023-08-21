#include "Fp32OPTAttention.h"
#include "operators.h"
#include "utils.h"
#include "utils_memalloc.h"

void test_Fp32OPTAttention() {
    const int num_heads = 12, embed_dim = 768, sqlen = 2, b = 1;
    MemoryAllocator mem_buf;

    Fp32OPTAttention::initialized_memory(get_opt_model_config(OPT_125M));
    Fp32OPTAttention attn =
        Fp32OPTAttention("FP32/models/OPT_125m/decoder/layer0/self_attn", get_opt_model_config(OPT_125M));

    Matrix3D<float> hidden_states(mem_buf.get_fpbuffer(embed_dim * sqlen), b, sqlen, embed_dim);
    hidden_states.load("assets/OPT/tests/attn/OPT_125m/Fp32_atten_input.bin");
    Matrix3D<float> attention_mask(mem_buf.get_fpbuffer(sqlen * sqlen), 1, sqlen, sqlen);
    attention_mask.load("assets/OPT/tests/attn/OPT_125m/Fp32_atten_mask.bin");
    struct Fp32OPTAttention_input input(hidden_states, attention_mask, 0);

    struct Fp32OPTAttention_output output = attn.forward(input);

    Matrix3D<float> attn_outputGT(mem_buf.get_fpbuffer(b * sqlen * embed_dim), b, sqlen, embed_dim);
    attn_outputGT.load("assets/OPT/tests/attn/OPT_125m/Fp32_atten_output.bin");

    bool success = check_two_equal(attn_outputGT.m_data, output.attn_output.m_data, b * sqlen * embed_dim);
    if (!success)
        std::cout << "Test of " << __func__ << ": Fail!" << std::endl;
    else
        std::cout << "-------- Test of " << __func__ << ": Passed! -------- " << std::endl;
}

void test_Fp32OPTAttention_1_3B() {
    const int embed_dim = 2048, sqlen = 2, b = 1;
    MemoryAllocator mem_buf;

    Fp32OPTAttention::initialized_memory(get_opt_model_config(OPT_1_3B));
    Fp32OPTAttention attn =
        Fp32OPTAttention("FP32/models/OPT_1.3B/decoder/layer0/self_attn", get_opt_model_config(OPT_1_3B));

    Matrix3D<float> hidden_states(mem_buf.get_fpbuffer(embed_dim * sqlen), b, sqlen, embed_dim);
    hidden_states.load("assets/OPT/tests/attn/OPT_1.3B/Fp32_atten_input.bin");
    Matrix3D<float> attention_mask(mem_buf.get_fpbuffer(sqlen * sqlen), 1, sqlen, sqlen);
    attention_mask.load("assets/OPT/tests/attn/OPT_1.3B/Fp32_atten_mask.bin");
    struct Fp32OPTAttention_input input(hidden_states, attention_mask, 0);

    struct Fp32OPTAttention_output output = attn.forward(input);

    Matrix3D<float> attn_outputGT(mem_buf.get_fpbuffer(b * sqlen * embed_dim), b, sqlen, embed_dim);
    attn_outputGT.load("assets/OPT/tests/attn/OPT_1.3B/Fp32_atten_output.bin");

    bool success = check_two_equal(attn_outputGT.m_data, output.attn_output.m_data, b * sqlen * embed_dim, 1e-5);
    if (!success)
        std::cout << "Test of " << __func__ << ": Fail!" << std::endl;
    else
        std::cout << "-------- Test of " << __func__ << ": Passed! -------- " << std::endl;
}

int main() {
    test_Fp32OPTAttention();
    test_Fp32OPTAttention_1_3B();
}
