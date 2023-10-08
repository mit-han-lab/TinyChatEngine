#include "Int4llamaAttention.h"
#include "operators.h"
#include "utils.h"

int NUM_THREAD = 8;

static void Int4LLaMAFreeMemory() {
    // Int4llamaAttention
    Int4llamaAttention llamaAttention;
    llamaAttention.free_cuda_memory();
}

void test_Int4llamaAttention() {
    const struct model_config llama7B = llama_7B;
    const int sqlen = 9, b = 1, embed_dim = llama7B.embed_dim, num_heads = llama7B.num_heads;

    Int4llamaAttention attn = Int4llamaAttention("INT4/models/LLaMA_7B_2_chat/decoder/layer0/self_attn", llama7B, 0);

    half* buffer_1;
    cudaMallocManaged(&buffer_1, sizeof(half) * embed_dim * sqlen * b);
    Matrix3D<half> hidden_states(buffer_1, b, sqlen, embed_dim);
    read_to_array_half("assets/llama/tests/atten/sqlen9/hidden_states_half.bin", hidden_states.m_data, hidden_states.length());
    half* buffer_2;
    cudaMallocManaged(&buffer_2, sizeof(half) * sqlen * sqlen);
    Matrix3D<half> attention_mask(buffer_2, 1, sqlen, sqlen);
    read_to_array_half("assets/llama/tests/atten/sqlen9/attention_mask_half.bin", attention_mask.m_data, attention_mask.length());

    attn.initialized_memory(llama7B);
    struct Int4llamaAttention_input input(hidden_states, attention_mask, 0);
    struct Int4llamaAttention_output output = attn.forward("INT4/models/LLaMA_7B_2_chat/decoder/layer0/self_attn", input);
    cudaDeviceSynchronize();

    half* buffer_3;
    cudaMallocManaged(&buffer_3, sizeof(half) * embed_dim * sqlen * b);
    Matrix3D<half> attn_outputGT(buffer_3, b, sqlen, embed_dim);
    read_to_array_half("assets/llama/tests/atten/sqlen9/attn_output_half.bin", attn_outputGT.m_data, attn_outputGT.length());

    half* buffer_4;
    cudaMallocManaged(&buffer_4, sizeof(half) * embed_dim * sqlen * b);
    Matrix3D<half> key_statesGT(buffer_4, num_heads, sqlen, embed_dim / num_heads);
    read_to_array_half("assets/llama/tests/atten/sqlen9/past_key_half.bin", key_statesGT.m_data, key_statesGT.length());

    half* buffer_5;
    cudaMallocManaged(&buffer_5, sizeof(half) * embed_dim * sqlen * b);
    Matrix3D<half> value_statesGT(buffer_5, num_heads, sqlen, embed_dim / num_heads);
    read_to_array_half("assets/llama/tests/atten/sqlen9/past_value_half.bin", value_statesGT.m_data, value_statesGT.length());

    bool success = check_two_equal_half_half(value_statesGT.m_data, output.past_key_value.second.m_data, value_statesGT.length());
    success &= check_two_equal_half_half(key_statesGT.m_data, output.past_key_value.first.m_data, key_statesGT.length());
    success &= check_two_equal_half_half(attn_outputGT.m_data, output.attn_output.m_data, attn_outputGT.length());
    if (!success)
        std::cout << "Test of " << __func__ << ": Fail!" << std::endl;
    else
        std::cout << "-------- Test of " << __func__ << ": Passed! -------- " << std::endl;

    // Free memory
    free_aligned_memory_gpu(buffer_1);
    free_aligned_memory_gpu(buffer_2);
    free_aligned_memory_gpu(buffer_3);
    free_aligned_memory_gpu(buffer_4);
    free_aligned_memory_gpu(buffer_5);
    Int4LLaMAFreeMemory();
}

void test_Int4llamaAttention_gen() {
    const struct model_config llama7B = llama_7B;
    const int sqlen = 1, b = 1, past_sqlen = 9, embed_dim = llama7B.embed_dim, num_heads = llama7B.num_heads,
              head_dim = embed_dim / num_heads;

    Int4llamaAttention attn = Int4llamaAttention("INT4/models/LLaMA_7B_2_chat/decoder/layer0/self_attn", llama7B, 0);

    half* buffer_1;
    cudaMallocManaged(&buffer_1, sizeof(half) * embed_dim * sqlen * b);
    Matrix3D<half> hidden_states(buffer_1, b, sqlen, embed_dim);
    read_to_array_half("assets/llama/tests/atten/sqlen1/hidden_states_half.bin", hidden_states.m_data, hidden_states.length());
    half* buffer_2;
    cudaMallocManaged(&buffer_2, sizeof(half) * sqlen * (sqlen + past_sqlen));
    Matrix3D<half> attention_mask(buffer_2, b, sqlen, sqlen + past_sqlen);
    read_to_array_half("assets/llama/tests/atten/sqlen1/attention_mask_half.bin", attention_mask.m_data, attention_mask.length());
    half* buffer_3;
    cudaMallocManaged(&buffer_3, sizeof(half) * num_heads * past_sqlen * embed_dim);
    Matrix3D<half> past_key(buffer_3, num_heads, past_sqlen, head_dim);
    read_to_array_half("assets/llama/tests/atten/sqlen9/past_key_half.bin", past_key.m_data, past_key.length());
    half* buffer_4;
    cudaMallocManaged(&buffer_4, sizeof(half) * num_heads * past_sqlen * embed_dim);
    Matrix3D<half> past_value(buffer_4, num_heads, past_sqlen, head_dim);
    read_to_array_half("assets/llama/tests/atten/sqlen9/past_value_half.bin", past_value.m_data, past_value.length());

    attn.initialized_memory(llama7B);
    struct Int4llamaAttention_input input(hidden_states, attention_mask, past_key, past_value, true, 0);
    struct Int4llamaAttention_output output = attn.forward("INT4/models/LLaMA_7B_2_chat/decoder/layer0/self_attn", input);
    cudaDeviceSynchronize();

    half* buffer_5;
    cudaMallocManaged(&buffer_5, sizeof(half) * embed_dim * sqlen * b);
    Matrix3D<half> attn_outputGT(buffer_5, b, sqlen, embed_dim);
    read_to_array_half("assets/llama/tests/atten/sqlen1/attn_output_half.bin", attn_outputGT.m_data, attn_outputGT.length());

    half* buffer_6;
    cudaMallocManaged(&buffer_6, sizeof(half) * (sqlen + past_sqlen) * embed_dim);
    Matrix3D<half> key_statesGT(buffer_6, num_heads, sqlen + past_sqlen, embed_dim / num_heads);
    read_to_array_half("assets/llama/tests/atten/sqlen1/past_key_half.bin", key_statesGT.m_data, key_statesGT.length());

    half* buffer_7;
    cudaMallocManaged(&buffer_7, sizeof(half) * (sqlen + past_sqlen) * embed_dim);
    Matrix3D<half> value_statesGT(buffer_7, num_heads, sqlen + past_sqlen, embed_dim / num_heads);
    read_to_array_half("assets/llama/tests/atten/sqlen1/past_value_half.bin", value_statesGT.m_data, value_statesGT.length());

    bool success = check_two_equal_half_half(value_statesGT.m_data, output.past_key_value.second.m_data, value_statesGT.length());
    success &= check_two_equal_half_half(key_statesGT.m_data, output.past_key_value.first.m_data, key_statesGT.length());
    success &= check_two_equal_half_half(attn_outputGT.m_data, output.attn_output.m_data, attn_outputGT.length());
    if (!success)
        std::cout << "Test of " << __func__ << ": Fail!" << std::endl;
    else
        std::cout << "-------- Test of " << __func__ << ": Passed! -------- " << std::endl;

    // Free memory
    free_aligned_memory_gpu(buffer_1);
    free_aligned_memory_gpu(buffer_2);
    free_aligned_memory_gpu(buffer_3);
    free_aligned_memory_gpu(buffer_4);
    free_aligned_memory_gpu(buffer_5);
    free_aligned_memory_gpu(buffer_6);
    free_aligned_memory_gpu(buffer_7);
    Int4LLaMAFreeMemory();
}

int main() {
    test_Int4llamaAttention();
    test_Int4llamaAttention_gen();
}
