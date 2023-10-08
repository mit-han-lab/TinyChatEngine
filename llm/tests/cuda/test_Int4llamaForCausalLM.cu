#include <chrono>
#include <cstring>

#include "Int4llamaForCausalLM.h"
#include "operators.h"
#include "utils.h"

int NUM_THREAD = 8;

static void Int4LLaMAFreeMemory() {
    // Int4LlamaForCausalLM
    Int4LlamaForCausalLM LlamaForCausalLM;
    LlamaForCausalLM.free_cuda_memory();

    // Int4llamaDecoder
    Int4llamaDecoder llamaDecoder;
    llamaDecoder.free_cuda_memory();

    // Int4llamaDecoderLayer
    Int4llamaDecoderLayer llamaDecoderLayer;
    llamaDecoderLayer.free_cuda_memory();

    // Int4llamaAttention
    Int4llamaAttention llamaAttention;
    llamaAttention.free_cuda_memory();
}

void test_Int4LlamaForCausalLM() {
    struct model_config config = get_opt_model_config(LLaMA_7B);
    const int voc_size = config.vocsize, sqlen = 9, b = 1;

    // reasoning phase: 1st run
    int* buffer_1;
    cudaMallocManaged(&buffer_1, sizeof(int) * sqlen);
    Matrix3D<int> input_ids(buffer_1, b, 1, sqlen);
    input_ids.load("assets/llama/tests/model/1st_input_ids.bin");
    struct Int4LlamaForCausalLM_input input_1st = {input_ids};

    Int4LlamaForCausalLM model = Int4LlamaForCausalLM("INT4/models/LLaMA_7B_2_chat", config);
    struct Int4LlamaForCausalLM_output output_1st = model.forward("INT4/models/LLaMA_7B_2_chat", input_1st);

    float* buffer_2;
    cudaMallocManaged(&buffer_2, sizeof(float) * b * sqlen * voc_size);
    Matrix3D<float> logits(buffer_2, b, sqlen, voc_size);
    logits.load("assets/llama/tests/model/1st_logits_cuda.bin");
    bool success = check_two_equal(output_1st.logits.m_data, logits.m_data, logits.length(), 1e-8);

    Profiler::getInstance().report();
    Profiler::getInstance().reset();

    // generating phase: 2nd run
    int* buffer_3;
    cudaMallocManaged(&buffer_3, sizeof(int) * sqlen);
    Matrix3D<int> input_ids_2nd(buffer_3, b, 1, 1);
    input_ids_2nd.load("assets/llama/tests/model/2nd_input_ids.bin");

    struct Int4LlamaForCausalLM_input input_2nd = {input_ids_2nd, output_1st.past_keys, output_1st.past_values};
    struct Int4LlamaForCausalLM_output output_2nd = model.forward("INT4/models/LLaMA_7B_2_chat", input_2nd);

    float* buffer_4;
    cudaMallocManaged(&buffer_4, sizeof(float) * b * 1 * voc_size);
    logits = Matrix3D<float>(buffer_4, b, 1, voc_size);
    logits.load("assets/llama/tests/model/2nd_logits_cuda.bin");

    success &= check_two_equal(output_2nd.logits.m_data, logits.m_data, logits.length(), 1e-8);

    Profiler::getInstance().report();

    if (!success)
        std::cout << "-------- Test of " << __func__ << ": Fail! -------- " << std::endl;
    else
        std::cout << "-------- Test of " << __func__ << ": Passed! -------- " << std::endl;

    // Free memory
    free_aligned_memory_gpu(buffer_1);
    free_aligned_memory_gpu(buffer_2);
    free_aligned_memory_gpu(buffer_3);
    free_aligned_memory_gpu(buffer_4);
    Int4LLaMAFreeMemory();
}

int main() { test_Int4LlamaForCausalLM(); }
