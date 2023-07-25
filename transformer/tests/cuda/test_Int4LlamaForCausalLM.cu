#include <chrono>
#include <cstring>

#include "Int4LlamaForCausalLM.h"
#include "operators.h"
#include "utils.h"
// #include "utils.cuh"
#include "utils_memalloc.cuh"

void test_Int4LlamaForCausalLM() {
    struct model_config config = get_opt_model_config(LLaMA_7B);
    const int num_heads = config.num_heads, embed_dim = config.embed_dim, voc_size = config.vocsize, sqlen = 9, b = 1;
    MemoryAllocator mem_buf;

    // reasoning phase: 1st run
    int* buffer_1;
    cudaMallocManaged(&buffer_1, sizeof(int) * sqlen);
    Matrix3D<int> input_ids(buffer_1, b, 1, sqlen);
    input_ids.load("assets/llama/tests/model/1st_input_ids.bin");
    struct Int4LlamaForCausalLM_input input_1st = {input_ids};

    Int4LlamaForCausalLM model = Int4LlamaForCausalLM("models/LLaMA_7B", config);
    struct Int4LlamaForCausalLM_output output_1st = model.forward(input_1st);

    float* buffer_2;
    cudaMallocManaged(&buffer_2, sizeof(float) * b * sqlen * voc_size);
    Matrix3D<float> logits(buffer_2, b, sqlen, voc_size);
    logits.load("assets/llama/tests/model/1st_logits.bin");
    bool success = check_two_equal(output_1st.logits.m_data, logits.m_data, logits.length(), 1e-8);

    float* buffer_3;
    cudaMallocManaged(&buffer_3, sizeof(float) * b * sqlen * embed_dim);
    Matrix3D<float> temp_key_value(buffer_3, num_heads, sqlen,
                                   embed_dim / num_heads);
    Profiler::getInstance().report();
    Profiler::getInstance().reset();

    // generating phase: 2nd run
    int* buffer_4;
    cudaMallocManaged(&buffer_4, sizeof(int) * sqlen);
    Matrix3D<int> input_ids_2nd(buffer_4, b, 1, 1);
    input_ids_2nd.load("assets/llama/tests/model/2nd_input_ids.bin");
    struct Int4LlamaForCausalLM_input input_2nd = {input_ids_2nd, output_1st.past_keys, output_1st.past_values};

    struct Int4LlamaForCausalLM_output output_2nd = model.forward(input_2nd);

    float* buffer_5;
    cudaMallocManaged(&buffer_5, sizeof(float) * b * 1 * voc_size);
    logits = Matrix3D<float>(buffer_5, b, 1, voc_size);
    logits.load("assets/llama/tests/model/2nd_logits.bin");
    success &= check_two_equal(output_2nd.logits.m_data, logits.m_data, logits.length(), 1e-8);

    Profiler::getInstance().report();
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
}

int main() { test_Int4LlamaForCausalLM(); }
