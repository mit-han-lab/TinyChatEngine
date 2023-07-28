#include <chrono>
#include <cstring>

#include "Fp32llamaForCausalLM.h"
#include "operators.h"
#include "utils.h"
#include "utils_memalloc.h"

void test_Fp32LlamaForCausalLM() {
    struct model_config config = get_opt_model_config(LLaMA_7B);
    const int num_heads = config.num_heads, embed_dim = config.embed_dim, sqlen = 9, b = 1,
              hidden_dim = config.hidden_dim;
    const int voc_size = config.vocsize, padding_idx = 1, num_layers = config.num_layers;
    MemoryAllocator mem_buf;

    // reasoning phase: 1st run
    Matrix3D<int> input_ids(mem_buf.get_intbuffer(sqlen), b, 1, sqlen);
    input_ids.load("assets/llama/tests/model/1st_input_ids.bin");
    struct Fp32LlamaForCausalLM_input input_1st = {input_ids};

    Fp32LlamaForCausalLM model = Fp32LlamaForCausalLM("models/LLaMA_7B", config);

    struct Fp32LlamaForCausalLM_output output_1st = model.forward(input_1st);

    Matrix3D<float> logits(mem_buf.get_fpbuffer(b * sqlen * voc_size), b, sqlen, voc_size);
    logits.load("assets/llama/tests/model/1st_logits.bin");
    // print_first_k_elelment("O", output_1st.logits.m_data, 20);
    // print_first_k_elelment("G", logits.m_data, 20);
    bool success = check_two_equal(output_1st.logits.m_data, logits.m_data, logits.length(), 1e-8);

    Matrix3D<float> temp_key_value(mem_buf.get_fpbuffer(b * sqlen * embed_dim), num_heads, sqlen,
                                   embed_dim / num_heads);
    Profiler::getInstance().report();
    Profiler::getInstance().reset();

    // generating phase: 2nd run
    Matrix3D<int> input_ids_2nd(mem_buf.get_intbuffer(sqlen), b, 1, 1);
    input_ids_2nd.load("assets/llama/tests/model/2nd_input_ids.bin");
    struct Fp32LlamaForCausalLM_input input_2nd = {input_ids_2nd, output_1st.past_keys, output_1st.past_values};

    struct Fp32LlamaForCausalLM_output output_2nd = model.forward(input_2nd);

    logits = Matrix3D<float>(mem_buf.get_fpbuffer(b * 1 * voc_size), b, 1, voc_size);
    logits.load("assets/llama/tests/model/2nd_logits.bin");
    // print_first_k_elelment("O", output_2nd.logits.m_data, 20);
    // print_first_k_elelment("G", logits.m_data, 20);
    success &= check_two_equal(output_2nd.logits.m_data, logits.m_data, logits.length(), 1e-8);

    Profiler::getInstance().report();
    if (!success)
        std::cout << "-------- Test of " << __func__ << ": Fail! -------- " << std::endl;
    else
        std::cout << "-------- Test of " << __func__ << ": Passed! -------- " << std::endl;
}

int main() { test_Fp32LlamaForCausalLM(); }
