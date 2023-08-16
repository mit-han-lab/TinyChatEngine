#include <cstring>

#include "Int4llamaDecoder.h"
#include "operators.h"
#include "utils.h"
#include "utils_memalloc.h"

void test_Decoder() {
    const struct model_config llama7B = llama_7B;
    const int sqlen = 9, b = 1, embed_dim = llama7B.embed_dim, num_heads = llama7B.num_heads,
              head_dim = embed_dim / num_heads, num_layers = llama7B.num_layers;
    MemoryAllocator mem_buf;

    Matrix3D<int> input_ids(mem_buf.get_intbuffer(sqlen), b, 1, sqlen);
    input_ids.load("assets/llama/tests/decoder/1st_input_ids.bin");
    struct Int4llamaDecoder_input input_1st = {input_ids};

    Int4llamaDecoder decoder = Int4llamaDecoder("models/LLaMA_7B/decoder/", llama7B);

    struct Int4llamaDecoder_output output_1st = decoder.forward(input_1st);

    // reasoning phase: 1st run
    Matrix3D<float> last_hidden_state1_GT(mem_buf.get_fpbuffer(b * sqlen * embed_dim), b, sqlen, embed_dim);
    last_hidden_state1_GT.load("assets/llama/tests/decoder/1st_last_hidden_state.bin");

    // print_first_k_elelment("output_1st.last_hidden_state", output_1st.last_hidden_state.m_data, 20);
    // print_first_k_elelment("last_hidden_state1_GT", last_hidden_state1_GT.m_data, 20);
    bool success = check_two_equal(output_1st.last_hidden_state.m_data, last_hidden_state1_GT.m_data,
                                   last_hidden_state1_GT.length(), 1e-8);

    Matrix3D<float> temp_key_value(mem_buf.get_fpbuffer(b * sqlen * embed_dim), num_heads, sqlen,
                                   embed_dim / num_heads);
    for (int i = 0; i < num_layers; i++) {
        std::string path = "assets/llama/tests/decoder/1st/past_key_value/key" + std::to_string(i) + ".bin";
        temp_key_value.load(path.c_str());
        success &=
            check_two_equal(output_1st.past_keys[i].m_data, temp_key_value.m_data, temp_key_value.length(), 1e-8);

        path = "assets/llama/tests/decoder/1st/past_key_value/value" + std::to_string(i) + ".bin";
        temp_key_value.load(path.c_str());
        success &=
            check_two_equal(output_1st.past_values[i].m_data, temp_key_value.m_data, temp_key_value.length(), 1e-8);
    }

    // generating phase: 2nd run
    Matrix3D<int> input_ids_2nd(mem_buf.get_intbuffer(sqlen), b, 1, 1);
    input_ids_2nd.load("assets/llama/tests/decoder/2nd/input_ids.bin");
    struct Int4llamaDecoder_input input_2nd = {input_ids_2nd, output_1st.past_keys, output_1st.past_values};

    struct Int4llamaDecoder_output output_2nd = decoder.forward(input_2nd);

    Matrix3D<float> last_hidden_state2_GT(mem_buf.get_fpbuffer(b * 1 * embed_dim), b, 1, embed_dim);
    last_hidden_state2_GT.load("assets/llama/tests/decoder/2nd/last_hidden_state.bin");
    success &= check_two_equal(output_2nd.last_hidden_state.m_data, last_hidden_state2_GT.m_data,
                               last_hidden_state2_GT.length(), 1e-8);

    temp_key_value = Matrix3D<float>(mem_buf.get_fpbuffer(b * (sqlen + 1) * embed_dim), num_heads, (sqlen + 1),
                                     embed_dim / num_heads);
    for (int i = 0; i < num_layers; i++) {
        std::string path = "assets/llama/tests/decoder/2nd/past_key_value/key" + std::to_string(i) + ".bin";
        temp_key_value.load(path.c_str());
        success &=
            check_two_equal(output_2nd.past_keys[i].m_data, temp_key_value.m_data, temp_key_value.length(), 1e-8);

        path = "assets/llama/tests/decoder/2nd/past_key_value/value" + std::to_string(i) + ".bin";
        temp_key_value.load(path.c_str());
        success &=
            check_two_equal(output_2nd.past_values[i].m_data, temp_key_value.m_data, temp_key_value.length(), 1e-8);
    }

    if (!success)
        std::cout << "-------- Test of " << __func__ << ": Fail! -------- " << std::endl;
    else
        std::cout << "-------- Test of " << __func__ << ": Passed! -------- " << std::endl;
}

int main() {
    // This tests are directly from fp32 and are not completed yet!
    // test_Decoder();
}
