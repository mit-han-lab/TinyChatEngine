#include "Int4llamaAttention.h"
#include "operators.h"
#include "utils.h"
#include "utils_memalloc.h"

void test_Int4llamaAttention() {
    const struct model_config llama7B = llama_7B;
    const int sqlen = 9, b = 1, embed_dim = llama7B.embed_dim, num_heads = llama7B.num_heads;

    MemoryAllocator mem_buf;

    Int4llamaAttention attn = Int4llamaAttention("models/LLaMA_7B/decoder/layer0/self_attn", llama7B);

    Matrix3D<float> hidden_states(mem_buf.get_fpbuffer(embed_dim * sqlen), b, sqlen, embed_dim);
    read_to_array("assets/llama/tests/atten/sqlen9/hidden_states.bin", hidden_states.m_data, b * sqlen * embed_dim);
    // print_first_k_elelment("hidden_states", hidden_states.m_data, 10);
    Matrix3D<float> attention_mask(mem_buf.get_fpbuffer(sqlen * sqlen), 1, sqlen, sqlen);
    read_to_array("assets/llama/tests/atten/sqlen9/attention_mask.bin", attention_mask.m_data, attention_mask.length());

    attn.initialized_memory(llama7B);
    struct Int4llamaAttention_input input(hidden_states, attention_mask, 0);

    struct Int4llamaAttention_output output = attn.forward(input);

    Matrix3D<float> attn_outputGT(mem_buf.get_fpbuffer(b * sqlen * embed_dim), b, sqlen, embed_dim);
    read_to_array("assets/llama/tests/atten/sqlen9/attn_output.bin", attn_outputGT.m_data, b * sqlen * embed_dim);
    Matrix3D<float> key_statesGT(mem_buf.get_fpbuffer(sqlen * embed_dim), num_heads, sqlen, embed_dim / num_heads);
    read_to_array("assets/llama/tests/atten/sqlen9/past_key.bin", key_statesGT.m_data, b * sqlen * embed_dim);
    Matrix3D<float> value_statesGT(mem_buf.get_fpbuffer(sqlen * embed_dim), num_heads, sqlen, embed_dim / num_heads);
    read_to_array("assets/llama/tests/atten/sqlen9/past_value.bin", value_statesGT.m_data, b * sqlen * embed_dim);

    bool success = check_two_equal(value_statesGT.m_data, output.past_key_value.second.m_data, value_statesGT.length());
    success &= check_two_equal(key_statesGT.m_data, output.past_key_value.first.m_data, key_statesGT.length());
    success &= check_two_equal(attn_outputGT.m_data, output.attn_output.m_data, attn_outputGT.length());
    if (!success)
        std::cout << "Test of " << __func__ << ": Fail!" << std::endl;
    else
        std::cout << "-------- Test of " << __func__ << ": Passed! -------- " << std::endl;
}

void test_Int4llamaAttention_gen() {
    const struct model_config llama7B = llama_7B;
    const int sqlen = 1, b = 1, past_sqlen = 9, embed_dim = llama7B.embed_dim, num_heads = llama7B.num_heads,
              head_dim = embed_dim / num_heads;

    MemoryAllocator mem_buf;

    Int4llamaAttention attn = Int4llamaAttention("models/LLaMA_7B/decoder/layer0/self_attn", llama7B);

    Matrix3D<float> hidden_states(mem_buf.get_fpbuffer(embed_dim * sqlen), b, sqlen, embed_dim);
    hidden_states.load("assets/llama/tests/atten/sqlen1/hidden_states.bin");
    Matrix3D<float> attention_mask(mem_buf.get_fpbuffer(sqlen * (sqlen + past_sqlen)), b, sqlen, sqlen + past_sqlen);
    attention_mask.load("assets/llama/tests/atten/sqlen1/attention_mask.bin");
    Matrix3D<float> past_key(mem_buf.get_fpbuffer(past_sqlen * embed_dim), num_heads, past_sqlen, head_dim);
    past_key.load("assets/llama/tests/atten/sqlen9/past_key.bin");
    Matrix3D<float> past_value(mem_buf.get_fpbuffer(past_sqlen * embed_dim), num_heads, past_sqlen, head_dim);
    past_value.load("assets/llama/tests/atten/sqlen9/past_value.bin");

    attn.initialized_memory(llama7B);
    struct Int4llamaAttention_input input(hidden_states, attention_mask, past_key, past_value, true, 0);

    struct Int4llamaAttention_output output = attn.forward(input);

    Matrix3D<float> attn_outputGT(mem_buf.get_fpbuffer(b * sqlen * embed_dim), b, sqlen, embed_dim);
    attn_outputGT.load("assets/llama/tests/atten/sqlen1/attn_output.bin");
    Matrix3D<float> key_statesGT(mem_buf.get_fpbuffer((sqlen + past_sqlen) * embed_dim), num_heads, sqlen + past_sqlen,
                                 embed_dim / num_heads);
    key_statesGT.load("assets/llama/tests/atten/sqlen1/past_key.bin");
    Matrix3D<float> value_statesGT(mem_buf.get_fpbuffer((sqlen + past_sqlen) * embed_dim), num_heads,
                                   sqlen + past_sqlen, embed_dim / num_heads);
    value_statesGT.load("assets/llama/tests/atten/sqlen1/past_value.bin");

    bool success = check_two_equal(value_statesGT.m_data, output.past_key_value.second.m_data, value_statesGT.length());
    success &= check_two_equal(key_statesGT.m_data, output.past_key_value.first.m_data, key_statesGT.length());
    success &= check_two_equal(attn_outputGT.m_data, output.attn_output.m_data, attn_outputGT.length());
    if (!success)
        std::cout << "Test of " << __func__ << ": Fail!" << std::endl;
    else
        std::cout << "-------- Test of " << __func__ << ": Passed! -------- " << std::endl;
}

int main() {
    // This tests are directly from fp32 and are not completed yet!
    // test_Int4llamaAttention();
    // test_Int4llamaAttention_gen();
}
