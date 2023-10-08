#include "Fp32OPTDecoderLayer.h"
#include "operators.h"
#include "utils.h"
#include "utils_memalloc.h"

int NUM_THREAD = 8;

void test_Fp32OPTDecoderLayer() {
    const int num_heads = 12, embed_dim = 768, sqlen = 2, b = 1;
    MemoryAllocator mem_buf;

    Fp32OPTDecoderLayer layer =
        Fp32OPTDecoderLayer("FP32/models/OPT_125m/decoder/layer0", get_opt_model_config(OPT_125M), 0);

    Matrix3D<float> hidden_states(mem_buf.get_fpbuffer(embed_dim * sqlen), b, sqlen, embed_dim);
    hidden_states.load("assets/OPT/tests/attn/OPT_125m/Fp32_layer_input.bin");
    Matrix3D<float> attention_mask(mem_buf.get_fpbuffer(sqlen * sqlen), 1, sqlen, sqlen);
    attention_mask.load("assets/OPT/tests/attn/OPT_125m/Fp32_layer_mask.bin");
    struct Fp32OPTDecoderLayer_input input(hidden_states, attention_mask);

    struct Fp32OPTDecoderLayer_output output = layer.forward(input);

    Matrix3D<float> attn_outputGT(mem_buf.get_fpbuffer(b * sqlen * embed_dim), b, sqlen, embed_dim);
    attn_outputGT.load("assets/OPT/tests/attn/OPT_125m/Fp32_layer_output.bin");

    bool success = check_two_equal(attn_outputGT.m_data, output.hidden_states.m_data, b * sqlen * embed_dim, 1e-4);
    if (!success)
        std::cout << "Test of " << __func__ << ": Fail!" << std::endl;
    else
        std::cout << "-------- Test of " << __func__ << ": Passed! -------- " << std::endl;
}

void test_Fp32OPTDecoderLayer_1_3B() {
    const int embed_dim = 2048, sqlen = 2, b = 1;
    MemoryAllocator mem_buf;

    Fp32OPTDecoderLayer layer =
        Fp32OPTDecoderLayer("FP32/models/OPT_1.3B/decoder/layer0", get_opt_model_config(OPT_1_3B), 0);

    Matrix3D<float> hidden_states(mem_buf.get_fpbuffer(embed_dim * sqlen), b, sqlen, embed_dim);
    hidden_states.load("assets/OPT/tests/attn/OPT_1.3B/Fp32_layer_input.bin");
    Matrix3D<float> attention_mask(mem_buf.get_fpbuffer(sqlen * sqlen), 1, sqlen, sqlen);
    attention_mask.load("assets/OPT/tests/attn/OPT_1.3B/Fp32_layer_mask.bin");
    struct Fp32OPTDecoderLayer_input input(hidden_states, attention_mask);

    struct Fp32OPTDecoderLayer_output output = layer.forward(input);

    Matrix3D<float> attn_outputGT(mem_buf.get_fpbuffer(b * sqlen * embed_dim), b, sqlen, embed_dim);
    attn_outputGT.load("assets/OPT/tests/attn/OPT_1.3B/Fp32_layer_output.bin");

    bool success = check_two_equal(attn_outputGT.m_data, output.hidden_states.m_data, b * sqlen * embed_dim, 1e-4);
    if (!success)
        std::cout << "Test of " << __func__ << ": Fail!" << std::endl;
    else
        std::cout << "-------- Test of " << __func__ << ": Passed! -------- " << std::endl;
}

int main() {
    test_Fp32OPTDecoderLayer();
    test_Fp32OPTDecoderLayer_1_3B();
}
