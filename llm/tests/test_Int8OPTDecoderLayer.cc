#include "Int8OPTDecoder.h"
#include "operators.h"
#include "utils.h"
#include "utils_memalloc.h"

void test_DecoderLayer_generate() {
    const int num_heads = 12, embed_dim = 768, sqlen = 108, b = 1, hidden_dim = 3072;
    MemoryAllocator mem_buf;

    Int8OPTDecoder decoder = Int8OPTDecoder("models/OPT_125m/decoder/", get_opt_model_config(OPT_125M));

    Matrix3D<float> hidden_states(mem_buf.get_fpbuffer(b * sqlen * embed_dim), b, sqlen, embed_dim);
    hidden_states.load("assets/OPT/tests/layer/OPT_125m/generate_layer0_hidden_states.bin");
    Matrix3D<float> attention_mask(mem_buf.get_fpbuffer(b * sqlen * sqlen), b, sqlen, sqlen);
    attention_mask.load("assets/OPT/tests/layer/OPT_125m/generate_layer0_attention_mask.bin");
    struct Int8OPTDecoderLayer_input input = {hidden_states, attention_mask};

    struct Int8OPTDecoderLayer_output output = decoder.layers[0].forward(input);

    Matrix3D<float> residualGT(mem_buf.get_fpbuffer(b * sqlen * embed_dim), b, sqlen, embed_dim);
    residualGT.load("assets/OPT/tests/layer/OPT_125m/generate_layer0_residual.bin");
    Matrix3D<int8_t> key_statesGT(mem_buf.get_int8buffer(output.past_key_value.first.length()),
                                  output.past_key_value.first.m_dim_x, output.past_key_value.first.m_dim_y,
                                  output.past_key_value.first.m_dim_z);
    key_statesGT.load("assets/OPT/tests/layer/OPT_125m/present_key.bin");
    Matrix3D<int8_t> value_statesGT(mem_buf.get_int8buffer(output.past_key_value.second.length()),
                                    output.past_key_value.second.m_dim_x, output.past_key_value.second.m_dim_y,
                                    output.past_key_value.second.m_dim_z);
    value_statesGT.load("assets/OPT/tests/layer/OPT_125m/present_value.bin");

    bool success = check_two_equal(residualGT.m_data, output.hidden_states.m_data, b * sqlen * embed_dim);
    success &= check_two_exact_equal(output.past_key_value.first.m_data, key_statesGT.m_data,
                                     output.past_key_value.first.length());
    success &= check_two_exact_equal(output.past_key_value.second.m_data, value_statesGT.m_data,
                                     output.past_key_value.second.length());
    if (!success)
        std::cout << "-------- Test of " << __func__ << ": Fail! -------- " << std::endl;
    else
        std::cout << "-------- Test of " << __func__ << ": Passed! -------- " << std::endl;
}

void test_DecoderLayer_generate_1_3B() {
    auto config = get_opt_model_config(OPT_1_3B);
    const int num_heads = config.num_heads, embed_dim = config.embed_dim, hidden_dim = config.hidden_dim;
    const int sqlen = 108, b = 1;
    MemoryAllocator mem_buf;

    Int8OPTDecoder decoder = Int8OPTDecoder("models/OPT_1.3B/decoder/", get_opt_model_config(OPT_1_3B));

    Matrix3D<float> hidden_states(mem_buf.get_fpbuffer(b * sqlen * embed_dim), b, sqlen, embed_dim);
    hidden_states.load("assets/OPT/tests/layer/OPT_1.3B/generate_layer0_hidden_states.bin");
    Matrix3D<float> attention_mask(mem_buf.get_fpbuffer(b * sqlen * sqlen), b, sqlen, sqlen);
    attention_mask.load("assets/OPT/tests/layer/OPT_1.3B/generate_layer0_attention_mask.bin");
    struct Int8OPTDecoderLayer_input input = {hidden_states, attention_mask};

    struct Int8OPTDecoderLayer_output output = decoder.layers[0].forward(input);

    Matrix3D<float> residualGT(mem_buf.get_fpbuffer(b * sqlen * embed_dim), b, sqlen, embed_dim);
    residualGT.load("assets/OPT/tests/layer/OPT_1.3B/generate_layer0_residual.bin");
    Matrix3D<int8_t> key_statesGT(mem_buf.get_int8buffer(output.past_key_value.first.length()),
                                  output.past_key_value.first.m_dim_x, output.past_key_value.first.m_dim_y,
                                  output.past_key_value.first.m_dim_z);
    key_statesGT.load("assets/OPT/tests/layer/OPT_1.3B/present_key.bin");
    Matrix3D<int8_t> value_statesGT(mem_buf.get_int8buffer(output.past_key_value.second.length()),
                                    output.past_key_value.second.m_dim_x, output.past_key_value.second.m_dim_y,
                                    output.past_key_value.second.m_dim_z);
    value_statesGT.load("assets/OPT/tests/layer/OPT_1.3B/present_value.bin");

    bool success = check_two_equal(residualGT.m_data, output.hidden_states.m_data, b * sqlen * embed_dim);
    success &= check_two_exact_equal(output.past_key_value.first.m_data, key_statesGT.m_data,
                                     output.past_key_value.first.length());
    success &= check_two_exact_equal(output.past_key_value.second.m_data, value_statesGT.m_data,
                                     output.past_key_value.second.length());
    if (!success)
        std::cout << "-------- Test of " << __func__ << ": Fail! -------- " << std::endl;
    else
        std::cout << "-------- Test of " << __func__ << ": Passed! -------- " << std::endl;
}

void test_DecoderLayer_generate_cache() {
    auto config = get_opt_model_config(OPT_125M);
    const int num_heads = config.num_heads, embed_dim = config.embed_dim, hidden_dim = config.hidden_dim;
    const int sqlen = 1, b = 1, past_len = 108, head_dim = embed_dim / num_heads;
    MemoryAllocator mem_buf;

    Int8OPTDecoder decoder = Int8OPTDecoder("models/OPT_125m/decoder/", get_opt_model_config(OPT_125M));

    int tgz = sqlen + past_len;
    Matrix3D<float> hidden_states(mem_buf.get_fpbuffer(b * sqlen * embed_dim), b, sqlen, embed_dim);
    hidden_states.load("assets/OPT/tests/layer/OPT_125m/test_cache_hidden_states.bin");
    Matrix3D<float> attention_mask(mem_buf.get_fpbuffer(b * sqlen * tgz), b, sqlen, tgz);
    attention_mask.load("assets/OPT/tests/layer/OPT_125m/test_cache_causal_attention_mask.bin");
    Matrix3D<int8_t> past_keys(mem_buf.get_int8buffer(b * past_len * embed_dim), num_heads, past_len, head_dim);
    past_keys.load("assets/OPT/tests/layer/OPT_125m/test_cache_past_key.bin");
    Matrix3D<int8_t> past_value(mem_buf.get_int8buffer(b * past_len * embed_dim), num_heads, past_len, head_dim);
    past_value.load("assets/OPT/tests/layer/OPT_125m/test_cache_past_value.bin");

    struct Int8OPTDecoderLayer_input input = {hidden_states, attention_mask, past_keys, past_value};

    struct Int8OPTDecoderLayer_output output = decoder.layers[0].forward(input);

    Matrix3D<float> residualGT(mem_buf.get_fpbuffer(b * sqlen * embed_dim), b, sqlen, embed_dim);
    residualGT.load("assets/OPT/tests/layer/OPT_125m/test_cache_residual.bin");
    // // print_first_k_elelment("output.hidden_states.m_data", output.hidden_states.m_data, 64);
    // // print_first_k_elelment("residualGT.m_data", residualGT.m_data, 64);
    Matrix3D<int8_t> key_statesGT(mem_buf.get_int8buffer(output.past_key_value.first.length()),
                                  output.past_key_value.first.m_dim_x, output.past_key_value.first.m_dim_y,
                                  output.past_key_value.first.m_dim_z);
    key_statesGT.load("assets/OPT/tests/layer/OPT_125m/test_present_key.bin");
    Matrix3D<int8_t> value_statesGT(mem_buf.get_int8buffer(output.past_key_value.second.length()),
                                    output.past_key_value.second.m_dim_x, output.past_key_value.second.m_dim_y,
                                    output.past_key_value.second.m_dim_z);
    value_statesGT.load("assets/OPT/tests/layer/OPT_125m/test_present_value.bin");

    bool success = check_two_equal(residualGT.m_data, output.hidden_states.m_data, b * sqlen * embed_dim);
    success &= check_two_exact_equal(output.past_key_value.first.m_data, key_statesGT.m_data,
                                     output.past_key_value.first.length());
    success &= check_two_exact_equal(output.past_key_value.second.m_data, value_statesGT.m_data,
                                     output.past_key_value.second.length());
    if (!success)
        std::cout << "-------- Test of " << __func__ << ": Fail! -------- " << std::endl;
    else
        std::cout << "-------- Test of " << __func__ << ": Passed! -------- " << std::endl;
}

void test_DecoderLayer_generate_cache_1_3B() {
    auto config = get_opt_model_config(OPT_1_3B);
    const int num_heads = config.num_heads, embed_dim = config.embed_dim, hidden_dim = config.hidden_dim;
    const int sqlen = 1, b = 1, past_len = 108, head_dim = embed_dim / num_heads;
    MemoryAllocator mem_buf;

    Int8OPTDecoder decoder = Int8OPTDecoder("models/OPT_1.3B/decoder/", get_opt_model_config(OPT_1_3B));

    int tgz = sqlen + past_len;
    Matrix3D<float> hidden_states(mem_buf.get_fpbuffer(b * sqlen * embed_dim), b, sqlen, embed_dim);
    hidden_states.load("assets/OPT/tests/layer/OPT_1.3B/test_cache_hidden_states.bin");
    Matrix3D<float> attention_mask(mem_buf.get_fpbuffer(b * sqlen * tgz), b, sqlen, tgz);
    attention_mask.load("assets/OPT/tests/layer/OPT_1.3B/test_cache_causal_attention_mask.bin");
    Matrix3D<int8_t> past_keys(mem_buf.get_int8buffer(b * past_len * embed_dim), num_heads, past_len, head_dim);
    past_keys.load("assets/OPT/tests/layer/OPT_1.3B/test_cache_past_key.bin");
    Matrix3D<int8_t> past_value(mem_buf.get_int8buffer(b * past_len * embed_dim), num_heads, past_len, head_dim);
    past_value.load("assets/OPT/tests/layer/OPT_1.3B/test_cache_past_value.bin");

    struct Int8OPTDecoderLayer_input input = {hidden_states, attention_mask, past_keys, past_value};

    struct Int8OPTDecoderLayer_output output = decoder.layers[0].forward(input);

    Matrix3D<float> residualGT(mem_buf.get_fpbuffer(b * sqlen * embed_dim), b, sqlen, embed_dim);
    residualGT.load("assets/OPT/tests/layer/OPT_1.3B/test_cache_residual.bin");
    Matrix3D<int8_t> key_statesGT(mem_buf.get_int8buffer(output.past_key_value.first.length()),
                                  output.past_key_value.first.m_dim_x, output.past_key_value.first.m_dim_y,
                                  output.past_key_value.first.m_dim_z);
    key_statesGT.load("assets/OPT/tests/layer/OPT_1.3B/test_present_key.bin");
    Matrix3D<int8_t> value_statesGT(mem_buf.get_int8buffer(output.past_key_value.second.length()),
                                    output.past_key_value.second.m_dim_x, output.past_key_value.second.m_dim_y,
                                    output.past_key_value.second.m_dim_z);
    value_statesGT.load("assets/OPT/tests/layer/OPT_1.3B/test_present_value.bin");

    bool success = check_two_equal(residualGT.m_data, output.hidden_states.m_data, b * sqlen * embed_dim);
    success &= check_two_exact_equal(output.past_key_value.first.m_data, key_statesGT.m_data,
                                     output.past_key_value.first.length());
    success &= check_two_exact_equal(output.past_key_value.second.m_data, value_statesGT.m_data,
                                     output.past_key_value.second.length());
    if (!success)
        std::cout << "-------- Test of " << __func__ << ": Fail! -------- " << std::endl;
    else
        std::cout << "-------- Test of " << __func__ << ": Passed! -------- " << std::endl;
}

void test_DecoderLayer() {
    auto config = get_opt_model_config(OPT_125M);
    const int num_heads = config.num_heads, embed_dim = config.embed_dim, hidden_dim = config.hidden_dim;
    const int sqlen = 512, b = 1, head_dim = embed_dim / num_heads;
    MemoryAllocator mem_buf;

    Int8OPTDecoder decoder = Int8OPTDecoder("models/OPT_125m/decoder/", get_opt_model_config(OPT_125M));

    Matrix3D<float> hidden_states(mem_buf.get_fpbuffer(b * sqlen * embed_dim), b, sqlen, embed_dim);
    hidden_states.load("assets/OPT/tests/layer/OPT_125m/Decoder_layer_hidden_states.bin");
    Matrix3D<float> attention_mask(mem_buf.get_fpbuffer(b * sqlen * sqlen), b, sqlen, sqlen);
    attention_mask.load("assets/OPT/tests/layer/OPT_125m/Decoder_attention_mask.bin");
    struct Int8OPTDecoderLayer_input input = {hidden_states, attention_mask};

    struct Int8OPTDecoderLayer_output output = decoder.layers[0].forward(input);

    Matrix3D<float> residualGT(mem_buf.get_fpbuffer(b * sqlen * embed_dim), b, sqlen, embed_dim);
    residualGT.load("assets/OPT/tests/layer/OPT_125m/Decoder_residual.bin");
    Matrix3D<int8_t> key_statesGT(mem_buf.get_int8buffer(output.past_key_value.first.length()),
                                  output.past_key_value.first.m_dim_x, output.past_key_value.first.m_dim_y,
                                  output.past_key_value.first.m_dim_z);
    key_statesGT.load("assets/OPT/tests/layer/OPT_125m/DecoderLayer_present_key.bin");
    Matrix3D<int8_t> value_statesGT(mem_buf.get_int8buffer(output.past_key_value.second.length()),
                                    output.past_key_value.second.m_dim_x, output.past_key_value.second.m_dim_y,
                                    output.past_key_value.second.m_dim_z);
    value_statesGT.load("assets/OPT/tests/layer/OPT_125m/DecoderLayer_present_value.bin");

    bool success = check_two_equal(residualGT.m_data, output.hidden_states.m_data, b * sqlen * embed_dim, 1e-5);
    success &= check_two_equal(output.past_key_value.first.m_data, key_statesGT.m_data,
                               output.past_key_value.first.length(), 1e-3);
    success &= check_two_equal(output.past_key_value.second.m_data, value_statesGT.m_data,
                               output.past_key_value.second.length(), 1e-4);
    // print_first_k_elelment("output.hidden_states.m_data", output.hidden_states.m_data, 10);
    // print_first_k_elelment("residualGT.m_data", residualGT.m_data, 10);
    if (!success)
        std::cout << "-------- Test of " << __func__ << ": Fail! -------- " << std::endl;
    else
        std::cout << "-------- Test of " << __func__ << ": Passed! -------- " << std::endl;
}

int main() {
    test_DecoderLayer();
    test_DecoderLayer_generate();
    test_DecoderLayer_generate_1_3B();
    test_DecoderLayer_generate_cache();
    test_DecoderLayer_generate_cache_1_3B();
}
