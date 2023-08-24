#include <chrono>
#include <cstring>

#include "OPTForCausalLM.h"
#include "operators.h"
#include "utils.h"
#include "utils_memalloc.h"

void test_OPTForCausalLM() {
    const int num_heads = 12, embed_dim = 768, sqlen = 108, b = 1, hidden_dim = 3072, voc_size = 50272, padding_idx = 1,
              num_layers = 12;
    MemoryAllocator mem_buf;

    // reasoning phase: 1st run
    Matrix3D<int> input_ids(mem_buf.get_intbuffer(sqlen), b, 1, sqlen);
    input_ids.load("assets/OPT/tests/causallm/OPT_125m/1st_input_ids.bin");
    struct OPTForCausalLM_input input_1st = {input_ids};

    OPTForCausalLM model = OPTForCausalLM("models/OPT_125m/", get_opt_model_config(OPT_125M));

    struct OPTForCausalLM_output output_1st = model.forward(input_1st);

    Matrix3D<float> logits(mem_buf.get_fpbuffer(b * sqlen * voc_size), b, sqlen, voc_size);
    logits.load("assets/OPT/tests/causallm/OPT_125m/1st_logits.bin");
    // print_first_k_elelment("O", output_1st.logits.m_data, 20);
    // print_first_k_elelment("G", logits.m_data, 20);
    bool success = check_two_equal(output_1st.logits.m_data, logits.m_data, logits.length(), 0.024);

    Matrix3D<int8_t> temp_key_value(mem_buf.get_int8buffer(b * sqlen * embed_dim), num_heads, sqlen,
                                    embed_dim / num_heads);
    Profiler::getInstance().report();
    Profiler::getInstance().reset();

    // generating phase: 2nd run
    Matrix3D<int> input_ids_2nd(mem_buf.get_intbuffer(sqlen), b, 1, 1);
    read_to_array("assets/OPT/tests/causallm/OPT_125m/2nd_input_ids.bin", input_ids_2nd.m_data, input_ids_2nd.length());
    struct OPTForCausalLM_input input_2nd = {input_ids_2nd, output_1st.past_keys, output_1st.past_values};

    struct OPTForCausalLM_output output_2nd = model.forward(input_2nd);

    logits = Matrix3D<float>(mem_buf.get_fpbuffer(b * 1 * voc_size), b, 1, voc_size);
    read_to_array("assets/OPT/tests/causallm/OPT_125m/2nd_logits.bin", logits.m_data, logits.length());
    // print_first_k_elelment("O", output_2nd.logits.m_data, 20);
    // print_first_k_elelment("G", logits.m_data, 20);
    success &= check_two_equal(output_2nd.logits.m_data, logits.m_data, logits.length(), 0.028);

    Profiler::getInstance().report();
    if (!success)
        std::cout << "-------- Test of " << __func__ << ": Fail! -------- " << std::endl;
    else
        std::cout << "-------- Test of " << __func__ << ": Passed! -------- " << std::endl;
}

// With more layers for 1.3B, the quantization errors (fp32 v.s. fp16) propagate even more. The whole pipeline seems
// correct, but the output (i.e., logits) is much different (MSE:0.506902). Below is decoded texts for different imp.
// fp16(native gpu), int8 (smoothquant), ours (fp32+int8). Instead of comparing raw outputs, we use hit rate to better
// validate the correctness of our implementation.

// fp16: 'I namepal play day is $ $ what of the other staff. she she was not seem like much lot, you,, it was a lot
// amount to meaire and\n the up car bill of the table, sheaire was to the bar and changehen up and when her for the a
// onto the lim truck\naulela already few of minutes servers in the with were her in the cake and clrian was j were the
// couple younger fun to they had cl feel. they helpedisted her into into the top of the cake.'

// int8: 'I namepal play day is $ $ what of the otherstaff. she she was not seem like much lot, some people, it was a
// lot price to meaud and\n the up car bill of the table, sheaire went to the bar and freshen up and then her for the on
// back the lim cart\namela already few of minutes servers in the with were her in the cake and therian was j were the
// couple younger fun to they had cl feel. they helpedisted her into into the top of the cake.'

// ours: ', namepal play day is $ $ what of the otherresses. the she was not seem like much lot, some people, it was a
// lot price to meaud and\n the up car bill of the table, sheaire went to the bar to changehen up and change her for the
// a back the back truck\namela already few of hours servers in the with were her in the cake and therian was j were the
// couple younger fun to they willing sure feel. they helpedisted her into into the top of the cake.'
void test_OPTForCausalLM_1_3B() {
    MemoryAllocator mem_buf;
    int sqlen = 108, b = 1;
    struct model_config config = get_opt_model_config(OPT_1_3B);
    int num_heads = config.num_heads, embed_dim = config.embed_dim, hidden_dim = config.hidden_dim,
        voc_size = config.vocsize, padding_idx = config.padding_idx, num_layers = config.num_layers;

    // reasoning phase: 1st run
    Matrix3D<int> input_ids(mem_buf.get_intbuffer(sqlen), b, 1, sqlen);
    input_ids.load("assets/OPT/tests/causallm/OPT_1.3B/1st_input_ids.bin");
    struct OPTForCausalLM_input input_1st = {input_ids};

    OPTForCausalLM model = OPTForCausalLM("models/OPT_1.3B", get_opt_model_config(OPT_1_3B));

    bool success = true;

    struct OPTForCausalLM_output output_1st = model.forward(input_1st);

    Matrix3D<float> logits(mem_buf.get_fpbuffer(b * sqlen * voc_size), b, sqlen, voc_size);
    logits.load("assets/OPT/tests/causallm/OPT_1.3B/1st_logits.bin");
    // print_first_k_elelment("O", output_1st.logits.m_data, 70, 50);
    // print_first_k_elelment("G", logits.m_data, 70, 50);
    success = check_two_equal(output_1st.logits.m_data, logits.m_data, logits.length(),
                              0.41);  // large error expected, see comments above

    Matrix3D<int> arg_max(mem_buf.get_intbuffer(sqlen), 1, 1, sqlen);
    arg_max_dim2(output_1st.logits, arg_max);
    Matrix3D<int> arg_maxGT(mem_buf.get_intbuffer(sqlen), 1, 1, sqlen);
    arg_maxGT.load("assets/OPT/tests/causallm/OPT_1.3B/1st_logits_arg_max_int8fp32.bin");

    int total_hit = 0;
    for (int i = 0; i < sqlen; i++)
        if (arg_maxGT.m_data[i] == arg_max.m_data[i]) total_hit++;
    float hit_rate = (float)total_hit / (float)sqlen;
    std::cout << "sqlen:" << sqlen << ", hits:" << total_hit << ", hit rate:" << hit_rate << std::endl;
    success &= hit_rate > 0.88;

    Profiler::getInstance().report();
    Profiler::getInstance().reset();

    // generating phase: 2nd run
    Matrix3D<int> input_ids_2nd(mem_buf.get_intbuffer(sqlen), b, 1, 1);
    input_ids_2nd.load("assets/OPT/tests/causallm/OPT_1.3B/2nd_input_ids.bin");
    struct OPTForCausalLM_input input_2nd = {input_ids_2nd, output_1st.past_keys, output_1st.past_values};

    struct OPTForCausalLM_output output_2nd = model.forward(input_2nd);

    logits = Matrix3D<float>(mem_buf.get_fpbuffer(b * 1 * voc_size), b, 1, voc_size);
    logits.load("assets/OPT/tests/causallm/OPT_1.3B/2nd_logits.bin");
    // print_first_k_elelment("O", output_2nd.logits.m_data, 20);
    // print_first_k_elelment("G", logits.m_data, 20);
    success &= check_two_equal(output_2nd.logits.m_data, logits.m_data, logits.length(), 1.77);

    Matrix3D<int> arg_max_2nd(mem_buf.get_intbuffer(sqlen), 1, 1, 1);
    arg_max_dim2(output_2nd.logits, arg_max_2nd);
    Matrix3D<int> arg_maxGT_2nd(mem_buf.get_intbuffer(sqlen), 1, 1, 1);
    arg_maxGT_2nd.load("assets/OPT/tests/causallm/OPT_1.3B/2nd_logits_arg_max_int8fp32.bin");

    total_hit = 0;
    for (int i = 0; i < 1; i++)
        if (arg_maxGT_2nd.m_data[i] == arg_max_2nd.m_data[i]) total_hit++;
    // std::cout << "arg_max_2nd.m_data:" << arg_max_2nd.m_data[0]  << ", arg_maxGT_2nd.m_data:" <<
    // arg_maxGT_2nd.m_data[0] << std::endl;
    hit_rate = (float)total_hit / (float)1;
    std::cout << "sqlen:" << 1 << ", hits:" << total_hit << ", hit rate:" << hit_rate << std::endl;
    success &= hit_rate > 0.99;

    Profiler::getInstance().report();
    if (!success)
        std::cout << "-------- Test of " << __func__ << ": Fail! -------- " << std::endl;
    else
        std::cout << "-------- Test of " << __func__ << ": Passed! -------- " << std::endl;
}

void test_OPTForCausalLM_6_7B() {
    MemoryAllocator mem_buf;
    int sqlen = 108, b = 1;
    struct model_config config = get_opt_model_config(OPT_6_7B);
    int num_heads = config.num_heads, embed_dim = config.embed_dim, hidden_dim = config.hidden_dim,
        voc_size = config.vocsize, padding_idx = config.padding_idx, num_layers = config.num_layers;

    // reasoning phase: 1st run
    Matrix3D<int> input_ids(mem_buf.get_intbuffer(sqlen), b, 1, sqlen);
    input_ids.load("assets/OPT/tests/causallm/OPT_6.7B/1st_input_ids.bin");
    struct OPTForCausalLM_input input_1st = {input_ids};

    // print_first_k_elelment("I", input_ids.m_data, 20);
    OPTForCausalLM model = OPTForCausalLM("models/OPT_6.7B", get_opt_model_config(OPT_6_7B));

    bool success = true;

    struct OPTForCausalLM_output output_1st = model.forward(input_1st);

    Matrix3D<float> logits(mem_buf.get_fpbuffer(b * sqlen * voc_size), b, sqlen, voc_size);
    logits.load("assets/OPT/tests/causallm/OPT_6.7B/1st_logits.bin");
    // print_first_k_elelment("O", output_1st.logits.m_data, 70, 50);
    // print_first_k_elelment("G", logits.m_data, 70, 50);
    // success = check_two_equal(output_1st.logits.m_data, logits.m_data, logits.length(),
    //                          0.507);  // large error expected, see comments above

    Matrix3D<int> arg_max(mem_buf.get_intbuffer(sqlen), 1, 1, sqlen);
    arg_max_dim2(output_1st.logits, arg_max);
    Matrix3D<int> arg_maxGT(mem_buf.get_intbuffer(sqlen), 1, 1, sqlen);
    arg_maxGT.load("assets/OPT/tests/causallm/OPT_6.7B/1st_logits_arg_max_int8fp32.bin");

    int total_hit = 0;
    for (int i = 0; i < sqlen; i++)
        if (arg_maxGT.m_data[i] == arg_max.m_data[i]) total_hit++;
    float hit_rate = (float)total_hit / (float)sqlen;
    // for (int i = 0; i < sqlen; i++) std::cout << arg_maxGT.m_data[i] << ",";
    // std::cout << std::endl;
    // for (int i = 0; i < sqlen; i++) std::cout << arg_max.m_data[i] << ",";
    // std::cout << std::endl;

    std::cout << "sqlen:" << sqlen << ", hits:" << total_hit << ", hit rate:" << hit_rate << std::endl;
    success &= hit_rate > 0.88;

    Profiler::getInstance().report();
    Profiler::getInstance().reset();

    // generating phase: 2nd run
    Matrix3D<int> input_ids_2nd(mem_buf.get_intbuffer(sqlen), b, 1, 1);
    input_ids_2nd.load("assets/OPT/tests/causallm/OPT_6.7B/2nd_input_ids.bin");
    struct OPTForCausalLM_input input_2nd = {input_ids_2nd, output_1st.past_keys, output_1st.past_values};

    struct OPTForCausalLM_output output_2nd = model.forward(input_2nd);

    logits = Matrix3D<float>(mem_buf.get_fpbuffer(b * 1 * voc_size), b, 1, voc_size);
    logits.load("assets/OPT/tests/causallm/OPT_6.7B/2nd_logits.bin");
    // print_first_k_elelment("O", output_2nd.logits.m_data, 20);
    // print_first_k_elelment("G", logits.m_data, 20);
    success &= check_two_equal(output_2nd.logits.m_data, logits.m_data, logits.length(), 0.184);

    Matrix3D<int> arg_max_2nd(mem_buf.get_intbuffer(sqlen), 1, 1, 1);
    arg_max_dim2(output_2nd.logits, arg_max_2nd);
    Matrix3D<int> arg_maxGT_2nd(mem_buf.get_intbuffer(sqlen), 1, 1, 1);
    arg_maxGT_2nd.load("assets/OPT/tests/causallm/OPT_6.7B/2nd_logits_arg_max_int8fp32.bin");

    total_hit = 0;
    for (int i = 0; i < 1; i++)
        if (arg_maxGT_2nd.m_data[i] == arg_max_2nd.m_data[i]) total_hit++;
    // std::cout << "arg_max_2nd.m_data:" << arg_max_2nd.m_data[0]  << ", arg_maxGT_2nd.m_data:" <<
    // arg_maxGT_2nd.m_data[0] << std::endl;
    hit_rate = (float)total_hit / (float)1;
    std::cout << "sqlen:" << 1 << ", hits:" << total_hit << ", hit rate:" << hit_rate << std::endl;
    success &= hit_rate > 0.99;

    Profiler::getInstance().report();
    if (!success)
        std::cout << "-------- Test of " << __func__ << ": Fail! -------- " << std::endl;
    else
        std::cout << "-------- Test of " << __func__ << ": Passed! -------- " << std::endl;
}

int main() {
    test_OPTForCausalLM();
    test_OPTForCausalLM_1_3B();
    test_OPTForCausalLM_6_7B();
}
