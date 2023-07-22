#include <cstring>

#include "Int4llamaDecoder.h"
#include "operators.h"
#include "utils.h"
#include "utils_memalloc.cuh"

void test_Decoder() {
    const struct model_config llama7B = llama_7B;
    const int sqlen = 9, b = 1, embed_dim = llama7B.embed_dim, num_heads = llama7B.num_heads, num_layers = llama7B.num_layers;
    MemoryAllocator mem_buf;

    int* buffer_1;
    cudaMallocManaged(&buffer_1, sizeof(int) * sqlen);
    Matrix3D<int> input_ids(buffer_1, b, 1, sqlen);
    input_ids.load("assets/llama/tests/decoder/1st_input_ids.bin");
    struct Int4llamaDecoder_input input_1st = {input_ids};
    
    Int4llamaDecoder decoder = Int4llamaDecoder("models/LLaMA_7B/decoder/", llama7B);
    struct Int4llamaDecoder_output output_1st = decoder.forward(input_1st);
    cudaDeviceSynchronize();

    // reasoning phase: 1st run
    float* buffer_2;
    cudaMallocManaged(&buffer_2, sizeof(float) * b * sqlen * embed_dim);
    Matrix3D<float> last_hidden_state1_GT(buffer_2, b, sqlen, embed_dim);
    last_hidden_state1_GT.load("assets/llama/tests/decoder/1st_last_hidden_state.bin");

    bool success = check_two_equal_float_half(last_hidden_state1_GT.m_data, output_1st.last_hidden_state.m_data,
                                   last_hidden_state1_GT.length());

    float* buffer_3;
    cudaMallocManaged(&buffer_3, sizeof(float) * b * sqlen * embed_dim);
    Matrix3D<float> temp_key_value(buffer_3, num_heads, sqlen, embed_dim / num_heads);
    for (int i = 0; i < num_layers; i++) {
        std::string path = "assets/llama/tests/decoder/1st/past_key_value/key" + std::to_string(i) + ".bin";
        temp_key_value.load(path.c_str());
        success &=
            check_two_equal_float_half(temp_key_value.m_data, output_1st.past_keys[i].m_data, temp_key_value.length());

        path = "assets/llama/tests/decoder/1st/past_key_value/value" + std::to_string(i) + ".bin";
        temp_key_value.load(path.c_str());
        success &=
            check_two_equal_float_half(temp_key_value.m_data, output_1st.past_values[i].m_data, temp_key_value.length());
    }


    // generating phase: 2nd run
    int* buffer_4;
    cudaMallocManaged(&buffer_4, sizeof(int) * sqlen);
    Matrix3D<int> input_ids_2nd(buffer_4, b, 1, 1);
    input_ids_2nd.load("assets/llama/tests/decoder/2nd/input_ids.bin");
    struct Int4llamaDecoder_input input_2nd = {input_ids_2nd, output_1st.past_keys, output_1st.past_values};

    struct Int4llamaDecoder_output output_2nd = decoder.forward(input_2nd);
    cudaDeviceSynchronize();


    float* buffer_5;
    cudaMallocManaged(&buffer_5, sizeof(float) * b * 1 * embed_dim);
    Matrix3D<float> last_hidden_state2_GT(buffer_5, b, 1, embed_dim);
    last_hidden_state2_GT.load("assets/llama/tests/decoder/2nd/last_hidden_state.bin");
    success &= check_two_equal_float_half(last_hidden_state2_GT.m_data, output_2nd.last_hidden_state.m_data,
                               last_hidden_state2_GT.length());

    float* buffer_6;
    cudaMallocManaged(&buffer_6, sizeof(float) * b * (sqlen + 1) * embed_dim);
    Matrix3D<float> temp_key_value_2nd(buffer_6, num_heads, (sqlen + 1), embed_dim / num_heads);
    for (int i = 0; i < num_layers; i++) {
        std::string path = "assets/llama/tests/decoder/2nd/past_key_value/key" + std::to_string(i) + ".bin";
        temp_key_value_2nd.load(path.c_str());
        success &=
            check_two_equal_float_half(temp_key_value_2nd.m_data, output_2nd.past_keys[i].m_data, temp_key_value_2nd.length());

        path = "assets/llama/tests/decoder/2nd/past_key_value/value" + std::to_string(i) + ".bin";
        temp_key_value_2nd.load(path.c_str());
        success &=
            check_two_equal_float_half(temp_key_value_2nd.m_data, output_2nd.past_values[i].m_data, temp_key_value_2nd.length());
    }

    if (!success)
        std::cout << "-------- Test of " << __func__ << ": Fail! -------- " << std::endl;
    else
        std::cout << "-------- Test of " << __func__ << ": Passed! -------- " << std::endl;

    // Free memory
    cudaFree(buffer_1);
    cudaFree(buffer_2);
    cudaFree(buffer_3);
    cudaFree(buffer_4);
    cudaFree(buffer_5);
    cudaFree(buffer_6);
}

int main() { test_Decoder(); }
