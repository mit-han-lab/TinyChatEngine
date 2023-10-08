#include "Fp32OPTForCausalLM.h"
#include "operators.h"
#include "utils.h"
#include "utils_memalloc.h"

int NUM_THREAD = 8;

void test_Fp32OPTForCausalLM() {
    struct model_config config = get_opt_model_config(OPT_1_3B);
    const int embed_dim = config.embed_dim, sqlen = 2, b = 1;
    const int voc_size = config.vocsize;
    MemoryAllocator mem_buf;

    Matrix3D<int> input_ids(mem_buf.get_intbuffer(sqlen), b, 1, sqlen);
    input_ids.load("assets/OPT/tests/attn/OPT_1.3B/Fp32_decoder_1st_input_ids.bin");
    struct Fp32OPTForCausalLM_input input_1st = {input_ids};

    Fp32OPTForCausalLM model = Fp32OPTForCausalLM("FP32/models/OPT_1.3B", get_opt_model_config(OPT_1_3B));

    struct Fp32OPTForCausalLM_output output_1st = model.forward(input_1st);

    // reasoning phase: 1st run
    Matrix3D<float> logits(mem_buf.get_fpbuffer(b * sqlen * voc_size), b, sqlen, voc_size);
    logits.load("assets/OPT/tests/attn/OPT_1.3B/Fp32_causallm_logits.bin");

    // print_first_k_elelment("logits", logits.m_data, 20);
    // print_first_k_elelment("output_1st.logits.m_data", output_1st.logits.m_data, 20);
    bool success = check_two_equal(output_1st.logits.m_data, logits.m_data, logits.length(), 1e-5);

    if (!success)
        std::cout << "Test of " << __func__ << ": Fail!" << std::endl;
    else
        std::cout << "-------- Test of " << __func__ << ": Passed! -------- " << std::endl;
}

int main() { test_Fp32OPTForCausalLM(); }
