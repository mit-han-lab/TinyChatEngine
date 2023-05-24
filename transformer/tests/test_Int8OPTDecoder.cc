#include <cstring>

#include "Int8OPTDecoder.h"
#include "operators.h"
#include "utils.h"

#define MAX_TEST_MEMORY_BUF 1024 * 1024 * 1024  // 1 GB
static char buffer[MAX_TEST_MEMORY_BUF];

class MemoryAllocator {
   public:
    MemoryAllocator() { this->counter = 0; }
    float* get_fpbuffer(int size) {
        int byte_size = size * sizeof(float);
        if (this->counter + byte_size > MAX_TEST_MEMORY_BUF) {
            throw("Memory allocation fails! Test case uses too much memory.");
        }
        int cur_counter = counter;
        this->counter += ((byte_size + 3) / 4) * 4;
        return (float*)&buffer[cur_counter];
    }
    int8_t* get_int8buffer(int size) {
        int byte_size = size * sizeof(int8_t);
        if (this->counter + byte_size > MAX_TEST_MEMORY_BUF) {
            throw("Memory allocation fails! Test case uses too much memory.");
        }
        int cur_counter = counter;
        this->counter += ((byte_size + 3) / 4) * 4;
        return (int8_t*)&buffer[cur_counter];
    }
    int* get_intbuffer(int size) {
        int byte_size = size * sizeof(int);
        if (this->counter + byte_size > MAX_TEST_MEMORY_BUF) {
            throw("Memory allocation fails! Test case uses too much memory.");
        }
        int cur_counter = counter;
        this->counter += ((byte_size + 3) / 4) * 4;
        return (int*)&buffer[cur_counter];
    }

   private:
    int counter;
};

// TODO: check if the errors here make sense
void test_Decoder_layers() {
    const int num_heads = 12, embed_dim = 768, sqlen = 108, b = 1, hidden_dim = 3072, voc_size = 50272, padding_idx = 1,
              num_layers = 12;
    bool sucess = true;
    MemoryAllocator mem_buf;

    Int8OPTDecoder decoder = Int8OPTDecoder("models/OPT_125m/decoder/", get_opt_model_config(OPT_125M));

    float* input_buf = mem_buf.get_fpbuffer(b * sqlen * embed_dim);
    Matrix3D<float> hidden_states(input_buf, b, sqlen, embed_dim);
    read_to_array("assets/tests/OPT_125m/generate_layer0_hidden_states.bin", hidden_states.m_data,
                  b * sqlen * embed_dim);
    Matrix3D<float> attention_mask(mem_buf.get_fpbuffer(b * sqlen * sqlen), b, sqlen, sqlen);
    read_to_array("assets/tests/OPT_125m/generate_layer0_attention_mask.bin", attention_mask.m_data, b * sqlen * sqlen);
    // print_first_k_elelment("hidden_states", hidden_states.m_data, 20);
    // print_first_k_elelment("attention_mask.m_data", attention_mask.m_data, 10);

    Matrix3D<float> residualGT(mem_buf.get_fpbuffer(b * sqlen * embed_dim), b, sqlen, embed_dim);
    Matrix3D<int8_t> temp_key_value(mem_buf.get_int8buffer(b * sqlen * embed_dim), num_heads, sqlen,
                                    embed_dim / num_heads);
    for (int i = 0; i < num_layers; i++) {
        struct Int8OPTDecoderLayer_input input = {hidden_states, attention_mask};
        struct Int8OPTDecoderLayer_output output = decoder.layers[i].forward(input);
        // struct Int8OPTDecoderLayer_output output = layer.forward(input);
        hidden_states = output.hidden_states;

        std::string path = "assets/tests/OPT_125m/hidden_states_layer" + std::to_string(i) + ".bin";
        read_to_array(path.c_str(), residualGT.m_data, b * sqlen * embed_dim);
        if (!check_two_equal(output.hidden_states.m_data, residualGT.m_data, residualGT.length(), 1.1e-3)) {
            std::cout << "hidden states:" << i << " failed" << std::endl;
            sucess = false;
        }
        path = "assets/tests/OPT_125m/decoder/decoder_1st_past_key" + std::to_string(i) + ".bin";
        read_to_array(path.c_str(), temp_key_value.m_data, temp_key_value.length());
        if (!check_two_equal(output.past_key_value.first.m_data, temp_key_value.m_data, temp_key_value.length(), 1.2)) {
            std::cout << "past key:" << i << " failed" << std::endl;
            sucess = false;
        }
        path = "assets/tests/OPT_125m/decoder/decoder_1st_past_value" + std::to_string(i) + ".bin";
        read_to_array(path.c_str(), temp_key_value.m_data, temp_key_value.length());
        if (!check_two_equal(output.past_key_value.second.m_data, temp_key_value.m_data, temp_key_value.length(),
                             1.75)) {
            std::cout << "past value:" << i << " failed" << std::endl;
            sucess = false;
        }
    }

    read_to_array("assets/tests/OPT_125m/hidden_states_layers_output.bin", residualGT.m_data, b * sqlen * embed_dim);
    sucess &= check_two_equal(residualGT.m_data, hidden_states.m_data, b * sqlen * embed_dim, 1.1e-3);
    if (!sucess)
        std::cout << "-------- Test of " << __func__ << ": Fail! -------- " << std::endl;
    else
        std::cout << "-------- Test of " << __func__ << ": Passed! -------- " << std::endl;
}

void test_Decoder_layers_1_3B() {
    auto config = get_opt_model_config(OPT_1_3B);
    const int num_heads = config.num_heads, embed_dim = config.embed_dim, sqlen = 108, b = 1,
              hidden_dim = config.hidden_dim, voc_size = config.vocsize, padding_idx = config.padding_idx,
              num_layers = config.num_layers;
    bool sucess = true;
    MemoryAllocator mem_buf;

    Int8OPTDecoder decoder = Int8OPTDecoder("models/OPT_1.3B/decoder/", get_opt_model_config(OPT_1_3B));

    float* input_buf = mem_buf.get_fpbuffer(b * sqlen * embed_dim);
    Matrix3D<float> hidden_states(input_buf, b, sqlen, embed_dim);
    read_to_array("assets/tests/OPT_1.3B/generate_layer0_hidden_states.bin", hidden_states.m_data,
                  b * sqlen * embed_dim);
    Matrix3D<float> attention_mask(mem_buf.get_fpbuffer(b * sqlen * sqlen), b, sqlen, sqlen);
    read_to_array("assets/tests/OPT_1.3B/generate_layer0_attention_mask.bin", attention_mask.m_data, b * sqlen * sqlen);
    // print_first_k_elelment("hidden_states", hidden_states.m_data, 20);
    // print_first_k_elelment("attention_mask.m_data", attention_mask.m_data, 10);

    Matrix3D<float> residualGT(mem_buf.get_fpbuffer(b * sqlen * embed_dim), b, sqlen, embed_dim);
    Matrix3D<int8_t> temp_key_value(mem_buf.get_int8buffer(b * sqlen * embed_dim), num_heads, sqlen,
                                    embed_dim / num_heads);
    for (int i = 0; i < num_layers; i++) {
        struct Int8OPTDecoderLayer_input input = {hidden_states, attention_mask};
        struct Int8OPTDecoderLayer_output output = decoder.layers[i].forward(input);
        // struct Int8OPTDecoderLayer_output output = layer.forward(input);
        hidden_states = output.hidden_states;

        std::string path = "assets/tests/OPT_1.3B/hidden_states_layer" + std::to_string(i) + ".bin";
        read_to_array(path.c_str(), residualGT.m_data, b * sqlen * embed_dim);
        // if (i == 4) {
        //     ;
        //     read_to_array(path.c_str(), hidden_states.m_data, b * sqlen * embed_dim);
        // }
        // printf("------------------- %d -----------------------\n", i);
        // print_MSE_max_diff(output.hidden_states.m_data, residualGT.m_data, residualGT.length());
        if (!check_two_equal(output.hidden_states.m_data, residualGT.m_data, residualGT.length())) {
            std::cout << "hidden states:" << i << " failed" << std::endl;
            sucess = false;
        }
        path = "assets/tests/OPT_1.3B/decoder/decoder_1st_past_key" + std::to_string(i) + ".bin";
        read_to_array(path.c_str(), temp_key_value.m_data, temp_key_value.length());
        if (!check_two_exact_equal(output.past_key_value.first.m_data, temp_key_value.m_data,
                                   temp_key_value.length())) {
            std::cout << "past key:" << i << " failed" << std::endl;
            sucess = false;
        }
        path = "assets/tests/OPT_1.3B/decoder/decoder_1st_past_value" + std::to_string(i) + ".bin";
        read_to_array(path.c_str(), temp_key_value.m_data, temp_key_value.length());
        if (!check_two_exact_equal(output.past_key_value.second.m_data, temp_key_value.m_data,
                                   temp_key_value.length())) {
            std::cout << "past value:" << i << " failed" << std::endl;
            sucess = false;
        }
    }

    read_to_array("assets/tests/OPT_1.3B/hidden_states_layers_output.bin", residualGT.m_data, b * sqlen * embed_dim);
    sucess &= check_two_equal(residualGT.m_data, hidden_states.m_data, b * sqlen * embed_dim, 1e-5);
    if (!sucess)
        std::cout << "-------- Test of " << __func__ << ": Fail! -------- " << std::endl;
    else
        std::cout << "-------- Test of " << __func__ << ": Passed! -------- " << std::endl;
}

void test_prepare_decoder_attention_mask() {
    const int num_heads = 12, embed_dim = 768, sqlen = 108, b = 1, hidden_dim = 3072, voc_size = 50272, padding_idx = 1,
              num_layers = 12;
    MemoryAllocator mem_buf;

    Int8OPTDecoder decoder = Int8OPTDecoder("models/OPT_125m/decoder/", get_opt_model_config(OPT_125M));

    // reasoning phase
    Matrix3D<float> causal_attention_mask = decoder.prepare_decoder_attention_mask(sqlen, 0);
    Matrix3D<float> causal_attention_maskGT(mem_buf.get_fpbuffer(b * sqlen * sqlen), b, sqlen, sqlen);
    read_to_array("assets/tests/decoder/generate_causal_attention_reason.bin", causal_attention_maskGT.m_data,
                  b * sqlen * sqlen);
    bool sucess = check_two_equal(causal_attention_maskGT.m_data, causal_attention_mask.m_data, b * sqlen * sqlen);

    // generating phase
    Matrix3D<float> causal_attention_mask_g = decoder.prepare_decoder_attention_mask(sqlen + 1, sqlen);
    Matrix3D<float> causal_attention_mask_gGT(mem_buf.get_fpbuffer(b * 1 * (sqlen + 1)), b, 1, sqlen + 1);
    read_to_array("assets/tests/decoder/generate_causal_attention_generate.bin", causal_attention_mask_gGT.m_data,
                  b * 1 * (sqlen + 1));
    sucess &= check_two_equal(causal_attention_mask_g.m_data, causal_attention_mask_gGT.m_data, b * 1 * (sqlen + 1));

    if (!sucess)
        std::cout << "-------- Test of " << __func__ << ": Fail! -------- " << std::endl;
    else
        std::cout << "-------- Test of " << __func__ << ": Passed! -------- " << std::endl;
}

void test_get_position_embed() {
    const int num_heads = 12, embed_dim = 768, sqlen = 108, b = 1, hidden_dim = 3072, voc_size = 50272, padding_idx = 1,
              num_layers = 12;
    MemoryAllocator mem_buf;

    Int8OPTDecoder decoder = Int8OPTDecoder("models/OPT_125m/decoder/", get_opt_model_config(OPT_125M));

    // reasoning phase
    Matrix3D<float> pos_embed = decoder.get_position_embed(sqlen, 0);
    assert(pos_embed.m_dim_y == sqlen);
    assert(pos_embed.m_dim_z == embed_dim);
    Matrix3D<float> pos_embedGT(mem_buf.get_fpbuffer(b * sqlen * embed_dim), b, sqlen, embed_dim);
    read_to_array("assets/tests/decoder/pos_embed_reasoning.bin", pos_embedGT.m_data, b * sqlen * embed_dim);
    bool sucess = check_two_equal(pos_embed.m_data, pos_embedGT.m_data, b * sqlen * embed_dim);

    // generating phase
    Matrix3D<float> pos_embed_g = decoder.get_position_embed(1, sqlen);
    Matrix3D<float> pos_embed_gGT(mem_buf.get_fpbuffer(b * 1 * embed_dim), b, 1, embed_dim);
    read_to_array("assets/tests/decoder/pos_embed_generate.bin", pos_embed_gGT.m_data, b * 1 * embed_dim);
    sucess &= check_two_equal(pos_embed_g.m_data, pos_embed_gGT.m_data, b * 1 * embed_dim);

    if (!sucess)
        std::cout << "-------- Test of " << __func__ << ": Fail! -------- " << std::endl;
    else
        std::cout << "-------- Test of " << __func__ << ": Passed! -------- " << std::endl;
}

void test_Decoder() {
    const int num_heads = 12, embed_dim = 768, sqlen = 108, b = 1, hidden_dim = 3072, voc_size = 50272, padding_idx = 1,
              num_layers = 12;
    MemoryAllocator mem_buf;

    Matrix3D<int> input_ids(mem_buf.get_intbuffer(sqlen), b, 1, sqlen);
    read_to_array("assets/tests/OPT_125m/decoder/decoder_1st_input_ids.bin", input_ids.m_data, b * sqlen);
    struct Int8OPTDecoder_input input_1st = {input_ids};

    Int8OPTDecoder decoder = Int8OPTDecoder("models/OPT_125m/decoder/", get_opt_model_config(OPT_125M));

    struct Int8OPTDecoder_output output_1st = decoder.forward(input_1st);

    // reasoning phase: 1st run
    Matrix3D<float> last_hidden_state1_GT(mem_buf.get_fpbuffer(b * sqlen * embed_dim), b, sqlen, embed_dim);
    read_to_array("assets/tests/OPT_125m/decoder/1st_last_hidden_state.bin", last_hidden_state1_GT.m_data,
                  last_hidden_state1_GT.length());
    // print_first_k_elelment("output_1st.last_hidden_state", output_1st.last_hidden_state.m_data, 20);
    // print_first_k_elelment("last_hidden_state1_GT", last_hidden_state1_GT.m_data, 20);
    bool sucess = check_two_equal(output_1st.last_hidden_state.m_data, last_hidden_state1_GT.m_data,
                                  last_hidden_state1_GT.length(), 0.02);

    Matrix3D<int8_t> temp_key_value(mem_buf.get_int8buffer(b * sqlen * embed_dim), num_heads, sqlen,
                                    embed_dim / num_heads);
    for (int i = 0; i < num_layers; i++) {
        std::string path = "assets/tests/OPT_125m/decoder/decoder_1st_past_key" + std::to_string(i) + ".bin";
        read_to_array(path.c_str(), temp_key_value.m_data, temp_key_value.length());
        // print_first_k_elelment("output_1st.past_keys[i].m_data", output_1st.past_keys[i].m_data, 20);
        // print_first_k_elelment("temp_key_value.m_data", temp_key_value.m_data, 20);
        sucess &= check_two_equal(output_1st.past_keys[i].m_data, temp_key_value.m_data, temp_key_value.length(), 1.2);

        path = "assets/tests/OPT_125m/decoder/decoder_1st_past_value" + std::to_string(i) + ".bin";
        read_to_array(path.c_str(), temp_key_value.m_data, temp_key_value.length());
        // print_first_k_elelment("output_1st.past_values[i].m_data", output_1st.past_values[i].m_data, 20);
        // print_first_k_elelment("temp_key_value.m_data", temp_key_value.m_data, 20);
        sucess &=
            check_two_equal(output_1st.past_values[i].m_data, temp_key_value.m_data, temp_key_value.length(), 1.8);
    }

    // generating phase: 2nd run
    Matrix3D<int> input_ids_2nd(mem_buf.get_intbuffer(sqlen), b, 1, 1);
    read_to_array("assets/tests/OPT_125m/decoder/2nd_input_ids.bin", input_ids_2nd.m_data, b * 1);
    struct Int8OPTDecoder_input input_2nd = {input_ids_2nd, output_1st.past_keys, output_1st.past_values};

    struct Int8OPTDecoder_output output_2nd = decoder.forward(input_2nd);

    Matrix3D<float> last_hidden_state2_GT(mem_buf.get_fpbuffer(b * 1 * embed_dim), b, 1, embed_dim);
    read_to_array("assets/tests/OPT_125m/decoder/2nd_last_hidden_state.bin", last_hidden_state2_GT.m_data,
                  last_hidden_state2_GT.length());
    // print_first_k_elelment("output_2nd.last_hidden_state", output_2nd.last_hidden_state.m_data, 20);
    // print_first_k_elelment("last_hidden_state2_GT.m_data", last_hidden_state2_GT.m_data, 20);
    sucess &= check_two_equal(output_2nd.last_hidden_state.m_data, last_hidden_state2_GT.m_data,
                              last_hidden_state2_GT.length(), 0.02);

    temp_key_value =
        Matrix3D<int8_t>(mem_buf.get_int8buffer(b * 1 * embed_dim), num_heads, (sqlen + 1), embed_dim / num_heads);
    for (int i = 0; i < num_layers; i++) {
        std::string path = "assets/tests/OPT_125m/decoder/decoder_2nd_past_key" + std::to_string(i) + ".bin";
        read_to_array(path.c_str(), temp_key_value.m_data, temp_key_value.length());
        // print_first_k_elelment("output_2nd.past_keys[i].m_data", output_2nd.past_keys[i].m_data, 20);
        // print_first_k_elelment("temp_key_value.m_data", temp_key_value.m_data, 20);
        sucess &= check_two_equal(output_2nd.past_keys[i].m_data, temp_key_value.m_data, temp_key_value.length(), 1.2);

        path = "assets/tests/OPT_125m/decoder/decoder_2nd_past_value" + std::to_string(i) + ".bin";
        read_to_array(path.c_str(), temp_key_value.m_data, temp_key_value.length());
        // print_first_k_elelment("output_2nd.past_values[i].m_data", output_2nd.past_values[i].m_data, 20);
        // print_first_k_elelment("temp_key_value.m_data", temp_key_value.m_data, 20);
        sucess &=
            check_two_equal(output_2nd.past_values[i].m_data, temp_key_value.m_data, temp_key_value.length(), 1.8);
    }

    if (!sucess)
        std::cout << "-------- Test of " << __func__ << ": Fail! -------- " << std::endl;
    else
        std::cout << "-------- Test of " << __func__ << ": Passed! -------- " << std::endl;
}

int main() {
    test_Decoder_layers();
    // test_Decoder_layers_1_3B();
    test_prepare_decoder_attention_mask();
    test_get_position_embed();
    test_Decoder();
}
