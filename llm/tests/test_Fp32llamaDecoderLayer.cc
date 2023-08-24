#include "Fp32llamaDecoderLayer.h"
#include "operators.h"
#include "utils.h"
#include "utils_memalloc.h"

void test_Fp32llamaDecoderLayer() {
    const struct model_config llama7B = llama_7B;
    const int sqlen = 9, b = 1, embed_dim = llama7B.embed_dim, num_heads = llama7B.num_heads;

    MemoryAllocator mem_buf;

    Fp32llamaDecoderLayer layer = Fp32llamaDecoderLayer("models/LLaMA_7B/decoder/layer0", llama7B, 0);

    Matrix3D<float> hidden_states(mem_buf.get_fpbuffer(embed_dim * sqlen), b, sqlen, embed_dim);
    hidden_states.load("assets/llama/tests/layer0/sqlen9/hidden_states.bin");
    // print_first_k_elelment("hidden_states", hidden_states.m_data, 10);
    Matrix3D<float> attention_mask(mem_buf.get_fpbuffer(sqlen * sqlen), 1, sqlen, sqlen);
    attention_mask.load("assets/llama/tests/layer0/sqlen9/attention_mask.bin");

    struct Fp32llamaDecoderLayer_input input(hidden_states, attention_mask);

    struct Fp32llamaDecoderLayer_output output = layer.forward(input);

    Matrix3D<float> outputGT(mem_buf.get_fpbuffer(b * sqlen * embed_dim), b, sqlen, embed_dim);
    outputGT.load("assets/llama/tests/layer0/sqlen9/output_hidden_states.bin");
    Matrix3D<float> key_statesGT(mem_buf.get_fpbuffer(sqlen * embed_dim), num_heads, sqlen, embed_dim / num_heads);
    key_statesGT.load("assets/llama/tests/layer0/sqlen9/present_key.bin");
    Matrix3D<float> value_statesGT(mem_buf.get_fpbuffer(sqlen * embed_dim), num_heads, sqlen, embed_dim / num_heads);
    value_statesGT.load("assets/llama/tests/layer0/sqlen9/present_value.bin");

    bool success = check_two_equal(value_statesGT.m_data, output.past_key_value.second.m_data, value_statesGT.length());
    success &= check_two_equal(key_statesGT.m_data, output.past_key_value.first.m_data, key_statesGT.length(), 1e-9);
    success &= check_two_equal(outputGT.m_data, output.hidden_states.m_data, outputGT.length());
    if (!success)
        std::cout << "Test of " << __func__ << ": Fail!" << std::endl;
    else
        std::cout << "-------- Test of " << __func__ << ": Passed! -------- " << std::endl;
}

void test_Fp32llamaDecoderLayer_gen() {
    const struct model_config llama7B = llama_7B;
    const int sqlen = 1, b = 1, past_sqlen = 9, embed_dim = llama7B.embed_dim, num_heads = llama7B.num_heads,
              head_dim = embed_dim / num_heads;
    const int tgz = (sqlen + past_sqlen);

    MemoryAllocator mem_buf;

    Fp32llamaDecoderLayer layer = Fp32llamaDecoderLayer("models/LLaMA_7B/decoder/layer0", llama7B, 0);

    Matrix3D<float> hidden_states(mem_buf.get_fpbuffer(embed_dim * sqlen), b, sqlen, embed_dim);
    hidden_states.load("assets/llama/tests/layer0/sqlen1/hidden_states.bin");
    Matrix3D<float> attention_mask(mem_buf.get_fpbuffer(sqlen * tgz), 1, sqlen, tgz);
    attention_mask.load("assets/llama/tests/layer0/sqlen1/attention_mask.bin");
    Matrix3D<float> past_key(mem_buf.get_fpbuffer(past_sqlen * embed_dim), num_heads, past_sqlen, head_dim);
    past_key.load("assets/llama/tests/layer0/sqlen1/past_key.bin");
    Matrix3D<float> past_value(mem_buf.get_fpbuffer(past_sqlen * embed_dim), num_heads, past_sqlen, head_dim);
    past_value.load("assets/llama/tests/layer0/sqlen1/past_value.bin");

    struct Fp32llamaDecoderLayer_input input(hidden_states, attention_mask, past_key, past_value);

    struct Fp32llamaDecoderLayer_output output = layer.forward(input);

    Matrix3D<float> outputGT(mem_buf.get_fpbuffer(b * sqlen * embed_dim), b, sqlen, embed_dim);
    outputGT.load("assets/llama/tests/layer0/sqlen1/output_hidden_states.bin");
    Matrix3D<float> key_statesGT(mem_buf.get_fpbuffer(tgz * embed_dim), num_heads, sqlen, embed_dim / num_heads);
    key_statesGT.load("assets/llama/tests/layer0/sqlen1/present_key.bin");
    Matrix3D<float> value_statesGT(mem_buf.get_fpbuffer(tgz * embed_dim), num_heads, tgz, embed_dim / num_heads);
    value_statesGT.load("assets/llama/tests/layer0/sqlen1/present_value.bin");

    bool success = check_two_equal(value_statesGT.m_data, output.past_key_value.second.m_data, value_statesGT.length());
    success &= check_two_equal(key_statesGT.m_data, output.past_key_value.first.m_data, key_statesGT.length(), 1e-9);
    success &= check_two_equal(outputGT.m_data, output.hidden_states.m_data, outputGT.length());
    if (!success)
        std::cout << "Test of " << __func__ << ": Fail!" << std::endl;
    else
        std::cout << "-------- Test of " << __func__ << ": Passed! -------- " << std::endl;
}

int main() {
    test_Fp32llamaDecoderLayer();
    test_Fp32llamaDecoderLayer_gen();
}
