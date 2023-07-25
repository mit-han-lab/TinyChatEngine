#include "Int4llamaDecoderLayer.h"
#include "operators.h"
#include "utils.h"
// #include "utils.cuh"
#include "utils_memalloc.cuh"

void test_Int4llamaDecoderLayer() {
    const struct model_config llama7B = llama_7B;
    const int sqlen = 9, b = 1, embed_dim = llama7B.embed_dim, num_heads = llama7B.num_heads;

    MemoryAllocator mem_buf;

    Int4llamaDecoderLayer layer = Int4llamaDecoderLayer("models/LLaMA_7B/decoder/layer0", llama7B, 0);

    float* buffer_1;
    cudaMallocManaged(&buffer_1, sizeof(float) * embed_dim * sqlen * b);
    Matrix3D<float> hidden_states_float(buffer_1, b, sqlen, embed_dim);
    hidden_states_float.load("assets/llama/tests/layer0/sqlen9/hidden_states.bin");

    half* buffer_1_half;
    cudaMallocManaged(&buffer_1_half, sizeof(half) * embed_dim * sqlen * b);
    Matrix3D<half> hidden_states(buffer_1_half, b, sqlen, embed_dim);
    int threadsPerBlock = 1024;
    int blocksPerGrid_1 =(b * sqlen * embed_dim + threadsPerBlock - 1) / threadsPerBlock;
    float2half<<<blocksPerGrid_1, threadsPerBlock>>>(buffer_1, buffer_1_half, b * sqlen * embed_dim);
    cudaDeviceSynchronize();


    float* buffer_2;
    cudaMallocManaged(&buffer_2, sizeof(float) * sqlen * sqlen);
    Matrix3D<float> attention_mask_float(buffer_2, 1, sqlen, sqlen);
    attention_mask_float.load("assets/llama/tests/layer0/sqlen9/attention_mask.bin");

    half* buffer_2_half;
    cudaMallocManaged(&buffer_2_half, sizeof(half) * sqlen * sqlen);
    Matrix3D<half> attention_mask(buffer_2_half, 1, sqlen, sqlen);
    int blocksPerGrid_2 =(sqlen * sqlen + threadsPerBlock - 1) / threadsPerBlock;
    float2half<<<blocksPerGrid_2, threadsPerBlock>>>(buffer_2, buffer_2_half, sqlen * sqlen);
    cudaDeviceSynchronize();

    struct Int4llamaDecoderLayer_input input(hidden_states, attention_mask);
    struct Int4llamaDecoderLayer_output output = layer.forward(input);
    cudaDeviceSynchronize();

    float* buffer_3;
    cudaMallocManaged(&buffer_3, sizeof(float) * embed_dim * sqlen * b);
    Matrix3D<float> outputGT(buffer_3, b, sqlen, embed_dim);
    outputGT.load("assets/llama/tests/layer0/sqlen9/output_hidden_states.bin");

    float* buffer_4;
    cudaMallocManaged(&buffer_4, sizeof(float) * embed_dim * sqlen * b);
    Matrix3D<float> key_statesGT(buffer_4, num_heads, sqlen, embed_dim / num_heads);
    key_statesGT.load("assets/llama/tests/layer0/sqlen9/present_key.bin");

    float* buffer_5;
    cudaMallocManaged(&buffer_5, sizeof(float) * embed_dim * sqlen * b);
    Matrix3D<float> value_statesGT(buffer_5, num_heads, sqlen, embed_dim / num_heads);
    value_statesGT.load("assets/llama/tests/layer0/sqlen9/present_value.bin");

    bool success = check_two_equal_float_half(value_statesGT.m_data, output.past_key_value.second.m_data, value_statesGT.length());
    success &= check_two_equal_float_half(key_statesGT.m_data, output.past_key_value.first.m_data, key_statesGT.length());
    success &= check_two_equal_float_half(outputGT.m_data, output.hidden_states.m_data, outputGT.length());
    if (!success)
        std::cout << "Test of " << __func__ << ": Fail!" << std::endl;
    else
        std::cout << "-------- Test of " << __func__ << ": Passed! -------- " << std::endl;

    // Free memory
    cudaFree(buffer_1);
    cudaFree(buffer_1_half);
    cudaFree(buffer_2);
    cudaFree(buffer_2_half);
    cudaFree(buffer_3);
    cudaFree(buffer_4);
    cudaFree(buffer_5);
}

void test_Int4llamaDecoderLayer_gen() {
    const struct model_config llama7B = llama_7B;
    const int sqlen = 1, b = 1, past_sqlen = 9, embed_dim = llama7B.embed_dim, num_heads = llama7B.num_heads,
              head_dim = embed_dim / num_heads;
    const int tgz = (sqlen + past_sqlen);

    MemoryAllocator mem_buf;

    Int4llamaDecoderLayer layer = Int4llamaDecoderLayer("models/LLaMA_7B/decoder/layer0", llama7B, 0);

    float* buffer_1;
    cudaMallocManaged(&buffer_1, sizeof(float) * embed_dim * sqlen * b);
    Matrix3D<float> hidden_states_float(buffer_1, b, sqlen, embed_dim);
    hidden_states_float.load("assets/llama/tests/layer0/sqlen1/hidden_states.bin");

    half* buffer_1_half;
    cudaMallocManaged(&buffer_1_half, sizeof(half) * embed_dim * sqlen * b);
    Matrix3D<half> hidden_states(buffer_1_half, b, sqlen, embed_dim);
    int threadsPerBlock = 1024;
    int blocksPerGrid_1 =(b * sqlen * embed_dim + threadsPerBlock - 1) / threadsPerBlock;
    float2half<<<blocksPerGrid_1, threadsPerBlock>>>(buffer_1, buffer_1_half, b * sqlen * embed_dim);
    cudaDeviceSynchronize();


    float* buffer_2;
    cudaMallocManaged(&buffer_2, sizeof(float) * sqlen * tgz);
    Matrix3D<float> attention_mask_float(buffer_2, 1, sqlen, tgz);
    attention_mask_float.load("assets/llama/tests/layer0/sqlen1/attention_mask.bin");

    half* buffer_2_half;
    cudaMallocManaged(&buffer_2_half, sizeof(half) * sqlen * tgz);
    Matrix3D<half> attention_mask(buffer_2_half, 1, sqlen, tgz);
    int blocksPerGrid_2 =(sqlen * tgz + threadsPerBlock - 1) / threadsPerBlock;
    float2half<<<blocksPerGrid_2, threadsPerBlock>>>(buffer_2, buffer_2_half, sqlen * tgz);
    cudaDeviceSynchronize();


    float* buffer_3;
    cudaMallocManaged(&buffer_3, sizeof(float) * num_heads * past_sqlen * embed_dim);
    Matrix3D<float> past_key_float(buffer_3, num_heads, past_sqlen, head_dim);
    past_key_float.load("assets/llama/tests/atten/sqlen9/past_key.bin");

    half* buffer_3_half;
    cudaMallocManaged(&buffer_3_half, sizeof(half) * num_heads * past_sqlen * embed_dim);
    Matrix3D<half> past_key(buffer_3_half, num_heads, past_sqlen, head_dim);
    int blocksPerGrid_3 =(num_heads * past_sqlen * embed_dim + threadsPerBlock - 1) / threadsPerBlock;
    float2half<<<blocksPerGrid_3, threadsPerBlock>>>(buffer_3, buffer_3_half, num_heads * past_sqlen * embed_dim);
    cudaDeviceSynchronize();


    float* buffer_4;
    cudaMallocManaged(&buffer_4, sizeof(float) * num_heads * past_sqlen * embed_dim);
    Matrix3D<float> past_value_float(buffer_4, num_heads, past_sqlen, head_dim);
    past_value_float.load("assets/llama/tests/atten/sqlen9/past_value.bin");

    half* buffer_4_half;
    cudaMallocManaged(&buffer_4_half, sizeof(half) * num_heads * past_sqlen * embed_dim);
    Matrix3D<half> past_value(buffer_4_half, num_heads, past_sqlen, head_dim);
    int blocksPerGrid_4 =(num_heads * past_sqlen * embed_dim + threadsPerBlock - 1) / threadsPerBlock;
    float2half<<<blocksPerGrid_4, threadsPerBlock>>>(buffer_4, buffer_4_half, num_heads * past_sqlen * embed_dim);
    cudaDeviceSynchronize();
    

    struct Int4llamaDecoderLayer_input input(hidden_states, attention_mask, past_key, past_value);
    struct Int4llamaDecoderLayer_output output = layer.forward(input);
    cudaDeviceSynchronize();


    float* buffer_5;
    cudaMallocManaged(&buffer_5, sizeof(float) * embed_dim * sqlen * b);
    Matrix3D<float> outputGT(buffer_5, b, sqlen, embed_dim);
    outputGT.load("assets/llama/tests/layer0/sqlen1/output_hidden_states.bin");

    float* buffer_6;
    cudaMallocManaged(&buffer_6, sizeof(float) * embed_dim * sqlen * b);
    Matrix3D<float> key_statesGT(buffer_6, num_heads, sqlen, embed_dim / num_heads);
    key_statesGT.load("assets/llama/tests/layer0/sqlen1/present_key.bin");

    float* buffer_7;
    cudaMallocManaged(&buffer_7, sizeof(float) * embed_dim * sqlen * b);
    Matrix3D<float> value_statesGT(buffer_7, num_heads, sqlen, embed_dim / num_heads);
    value_statesGT.load("assets/llama/tests/layer0/sqlen1/present_value.bin");


    bool success = check_two_equal_float_half(value_statesGT.m_data, output.past_key_value.second.m_data, value_statesGT.length());
    success &= check_two_equal_float_half(key_statesGT.m_data, output.past_key_value.first.m_data, key_statesGT.length());
    success &= check_two_equal_float_half(outputGT.m_data, output.hidden_states.m_data, outputGT.length());
    if (!success)
        std::cout << "Test of " << __func__ << ": Fail!" << std::endl;
    else
        std::cout << "-------- Test of " << __func__ << ": Passed! -------- " << std::endl;

    // Free memory
    cudaFree(buffer_1);
    cudaFree(buffer_1_half);
    cudaFree(buffer_2);
    cudaFree(buffer_2_half);
    cudaFree(buffer_3);
    cudaFree(buffer_3_half);
    cudaFree(buffer_4);
    cudaFree(buffer_4_half);
    cudaFree(buffer_5);
    cudaFree(buffer_6);
    cudaFree(buffer_7);
}

int main() {
    test_Int4llamaDecoderLayer();
    test_Int4llamaDecoderLayer_gen();
}
