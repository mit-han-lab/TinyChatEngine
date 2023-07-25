#include <cmath>

#include "operators.h"
#include "utils.h"
#include "utils.cuh"
#include "../utils_memalloc.h"
#include "linear.cuh"

void test_LlamaRMSNorm_cuda() {
    const int b = 1, m = 108, n = 768;
    MemoryAllocator mem_buf;

    float *intput_arr = mem_buf.get_fpbuffer(b * m * n);
    float *weight_arr = mem_buf.get_fpbuffer(b * n);
    float *bias_arr = mem_buf.get_fpbuffer(b * n);
    float *output_arr = mem_buf.get_fpbuffer(b * m * n);
    float *GToutput_arr = mem_buf.get_fpbuffer(b * m * n);

    Matrix3D<float> input(intput_arr, b, m, n);
    Matrix3D<float> weight(weight_arr, b, 1, n);
    Matrix3D<float> bias(bias_arr, b, 1, n);
    Matrix3D<float> output(output_arr, b, m, n);
    Matrix3D<float> GToutput(GToutput_arr, b, m, n);

    input.load("assets/OPT/tests/ops/OPT_125m/decoder/final_layer_norm_hidden_states.bin");
    GToutput.load("assets/OPT/tests/ops/OPT_125m/decoder/final_layer_norm_output.bin");

    struct LayerNorm_params op_params = {weight, bias};

    LayerNorm test_op = LayerNorm(op_params);
    load_LayerNorm(test_op, "models/OPT_125m/decoder/final_layer_norm/");

    test_op.forward(input, output);

    bool success = check_two_equal(output_arr, GToutput_arr, b * m * n);
    if (!success)
        std::cout << "-------- Test of " << __func__ << ": Fail! -------- " << std::endl;
    else
        std::cout << "-------- Test of " << __func__ << ": Passed! -------- " << std::endl;
}

void test_W8A8B8O8LinearReLU() {
    const int b = 1, m = 108, k = 768, n = 3072;
    const float alpha = 0.0005035400390625, beta = 0.02130126953125;
    MemoryAllocator mem_buf;

    int8_t *intput_arr = mem_buf.get_int8buffer(b * m * k);
    int8_t *weight_arr = mem_buf.get_int8buffer(b * k * n);
    int8_t *biasint8_arr = mem_buf.get_int8buffer(b * n);
    int8_t *output_arr = mem_buf.get_int8buffer(b * m * n);
    int8_t *GToutput_arr = mem_buf.get_int8buffer(b * m * n);

    Matrix3D<int8_t> input(intput_arr, b, m, k);
    Matrix3D<int8_t> weight(weight_arr, b, n, k);
    Matrix3D<int8_t> bias(biasint8_arr, b, 1, n);
    Matrix3D<int8_t> output(output_arr, b, m, n);
    Matrix3D<int8_t> GToutput(GToutput_arr, b, m, n);

    input.load("assets/OPT/tests/ops/OPT_125m/W8A8B8O8LinearReLU_x.bin");
    GToutput.load("assets/OPT/tests/ops/OPT_125m/W8A8B8O8LinearReLU_y.bin");

    struct W8A8B8O8LinearReLU_params op_params = {weight, bias, alpha, beta};

    W8A8B8O8LinearReLU test_op = W8A8B8O8LinearReLU(op_params);
    load_W8A8B8O8LinearReLU_params(test_op, "models/OPT_125m/decoder/layer0/fc1/");

    test_op.forward(input, output);

    bool success = check_two_exact_equal(output_arr, GToutput_arr, m * n);
    if (!success)
        std::cout << "-------- Test of " << __func__ << ": Fail! -------- " << std::endl;
    else
        std::cout << "-------- Test of " << __func__ << ": Passed! -------- " << std::endl;
}

void test_W8A8B8O8LinearReLU_1_3B() {
    const int b = 1, m = 108, k = 2048, n = 8192;
    const float alpha = 0.0025501251220703125, beta = 0.0106048583984375;
    MemoryAllocator mem_buf;

    int8_t *intput_arr = mem_buf.get_int8buffer(b * m * k);
    int8_t *weight_arr = mem_buf.get_int8buffer(b * k * n);
    int8_t *biasint8_arr = mem_buf.get_int8buffer(b * n);
    int8_t *output_arr = mem_buf.get_int8buffer(b * m * n);
    int8_t *GToutput_arr = mem_buf.get_int8buffer(b * m * n);

    Matrix3D<int8_t> input(intput_arr, b, m, k);
    Matrix3D<int8_t> weight(weight_arr, b, n, k);
    Matrix3D<int8_t> bias(biasint8_arr, b, 1, n);
    Matrix3D<int8_t> output(output_arr, b, m, n);
    Matrix3D<int8_t> GToutput(GToutput_arr, b, m, n);

    input.load("assets/OPT/tests/ops/OPT_1.3B/W8A8B8O8LinearReLU_x.bin");
    GToutput.load("assets/OPT/tests/ops/OPT_1.3B/W8A8B8O8LinearReLU_y.bin");

    struct W8A8B8O8LinearReLU_params op_params = {weight, bias, alpha, beta};

    W8A8B8O8LinearReLU test_op = W8A8B8O8LinearReLU(op_params);
    load_W8A8B8O8LinearReLU_params(test_op, "models/OPT_1.3B/decoder/layer0/fc1/");

    test_op.forward(input, output);

    bool success = check_two_exact_equal(output_arr, GToutput_arr, m * n);
    if (!success)
        std::cout << "-------- Test of " << __func__ << ": Fail! -------- " << std::endl;
    else
        std::cout << "-------- Test of " << __func__ << ": Passed! -------- " << std::endl;
}

void test_W8A8BFP32OFP32Linear() {
    const int b = 1, m = 512, k = 768, n = 768;
    const float alpha = 0.00004565715789794922;
    MemoryAllocator mem_buf;

    int8_t *intput_arr = mem_buf.get_int8buffer(b * m * k);
    int8_t *weight_arr = mem_buf.get_int8buffer(b * k * n);
    float *bias_arr = mem_buf.get_fpbuffer(b * n);
    float *output_arr = mem_buf.get_fpbuffer(b * m * n);
    float *GToutput_arr = mem_buf.get_fpbuffer(b * m * n);

    Matrix3D<int8_t> input(intput_arr, b, m, k);
    Matrix3D<int8_t> weight(weight_arr, b, n, k);
    Matrix3D<float> bias(bias_arr, b, 1, n);
    Matrix3D<float> output(output_arr, b, m, n);
    Matrix3D<float> GToutput(GToutput_arr, b, m, n);

    input.load("assets/OPT/tests/ops/OPT_125m/W8A8BFP32OFP32Linear_x.bin");
    GToutput.load("assets/OPT/tests/ops/OPT_125m/W8A8BFP32OFP32Linear_y.bin");

    struct W8A8BFP32OFP32Linear_params op_params = {weight, bias, alpha};

    auto test_op = W8A8BFP32OFP32Linear(op_params);
    load_W8A8BFP32OFP32Linear_params(test_op, "models/OPT_125m/decoder/layer0/self_attn/out_proj/");
    test_op.forward(input, output);

    bool success = check_two_equal(output_arr, GToutput_arr, b * m * n);
    if (!success)
        std::cout << "-------- Test of " << __func__ << ": Fail! -------- " << std::endl;
    else
        std::cout << "-------- Test of " << __func__ << ": Passed! -------- " << std::endl;
}

void test_W8A8BFP32OFP32Linear_1_3B() {
    const int b = 1, m = 108, k = 2048, n = 2048;
    const float alpha = 0.00012445449829101562;
    MemoryAllocator mem_buf;

    int8_t *intput_arr = mem_buf.get_int8buffer(b * m * k);
    int8_t *weight_arr = mem_buf.get_int8buffer(b * k * n);
    float *bias_arr = mem_buf.get_fpbuffer(b * n);
    float *output_arr = mem_buf.get_fpbuffer(b * m * n);
    float *GToutput_arr = mem_buf.get_fpbuffer(b * m * n);

    Matrix3D<int8_t> input(intput_arr, b, m, k);
    Matrix3D<int8_t> weight(weight_arr, b, n, k);
    Matrix3D<float> bias(bias_arr, b, 1, n);
    Matrix3D<float> output(output_arr, b, m, n);
    Matrix3D<float> GToutput(GToutput_arr, b, m, n);

    input.load("assets/OPT/tests/ops/OPT_1.3B/W8A8BFP32OFP32Linear_x.bin");
    GToutput.load("assets/OPT/tests/ops/OPT_1.3B/W8A8BFP32OFP32Linear_y.bin");

    struct W8A8BFP32OFP32Linear_params op_params = {weight, bias, alpha};

    auto test_op = W8A8BFP32OFP32Linear(op_params);
    load_W8A8BFP32OFP32Linear_params(test_op, "models/OPT_1.3B/decoder/layer0/self_attn/out_proj/");
    test_op.forward(input, output);

    bool success = check_two_equal(output_arr, GToutput_arr, b * m * n);
    if (!success)
        std::cout << "-------- Test of " << __func__ << ": Fail! -------- " << std::endl;
    else
        std::cout << "-------- Test of " << __func__ << ": Passed! -------- " << std::endl;
}

void test_W8A8B8O8Linear() {
    const int b = 1, m = 108, k = 768, n = 768;
    MemoryAllocator mem_buf;

    int8_t *intput_arr = mem_buf.get_int8buffer(b * m * k);
    int8_t *weight_arr = mem_buf.get_int8buffer(b * k * n);
    int8_t *weightGT_arr = mem_buf.get_int8buffer(b * k * n);
    int8_t *biasint8_arr = mem_buf.get_int8buffer(b * n);
    int8_t *biasint8GT_arr = mem_buf.get_int8buffer(b * n);
    int32_t *biasint32_arr = mem_buf.get_intbuffer(b * n);
    int8_t *output_arr = mem_buf.get_int8buffer(b * m * n);
    int8_t *GToutput_arr = mem_buf.get_int8buffer(b * m * n);

    Matrix3D<int8_t> input(intput_arr, b, m, k);
    Matrix3D<int8_t> weight(weight_arr, b, n, k);
    Matrix3D<int8_t> bias(biasint8_arr, b, 1, n);
    Matrix3D<int8_t> output(output_arr, b, m, n);
    Matrix3D<int8_t> GToutput(GToutput_arr, b, m, n);

    input.load("assets/OPT/tests/ops/OPT_125m/W8A8B8O8Linear_x.bin");
    GToutput.load("assets/OPT/tests/ops/OPT_125m/W8A8B8O8Linear_y.bin");

    struct W8A8B8O8Linear_params op_params = {weight, bias};

    W8A8B8O8Linear test_op = W8A8B8O8Linear(op_params);
    load_W8A8B8O8Linear_params(test_op, "models/OPT_125m/decoder/layer0/self_attn/q_proj/");

    test_op.forward(input, output);

    bool success = check_two_exact_equal(output_arr, GToutput_arr, b * m * n);
    if (!success)
        std::cout << "-------- Test of " << __func__ << ": Fail! -------- " << std::endl;
    else
        std::cout << "-------- Test of " << __func__ << ": Passed! -------- " << std::endl;
}

void test_W8A8B8O8Linear_1_3B() {
    const int b = 1, m = 108, k = 2048, n = 2048;
    MemoryAllocator mem_buf;

    int8_t *intput_arr = mem_buf.get_int8buffer(b * m * k);
    int8_t *weight_arr = mem_buf.get_int8buffer(b * k * n);
    int8_t *biasint8_arr = mem_buf.get_int8buffer(b * n);
    int8_t *output_arr = mem_buf.get_int8buffer(b * m * n);
    int8_t *GToutput_arr = mem_buf.get_int8buffer(b * m * n);

    Matrix3D<int8_t> input(intput_arr, b, m, k);
    Matrix3D<int8_t> weight(weight_arr, b, n, k);
    Matrix3D<int8_t> bias(biasint8_arr, b, 1, n);
    Matrix3D<int8_t> output(output_arr, b, m, n);
    Matrix3D<int8_t> GToutput(GToutput_arr, b, m, n);

    input.load("assets/OPT/tests/ops/OPT_1.3B/W8A8B8O8Linear_x.bin");
    GToutput.load("assets/OPT/tests/ops/OPT_1.3B/W8A8B8O8Linear_y.bin");

    struct W8A8B8O8Linear_params op_params = {weight, bias};

    W8A8B8O8Linear test_op = W8A8B8O8Linear(op_params);
    load_W8A8B8O8Linear_params(test_op, "models/OPT_1.3B/decoder/layer0/self_attn/q_proj/");

    test_op.forward(input, output);

    bool success = check_two_exact_equal(output_arr, GToutput_arr, b * m * n);
    if (!success)
        std::cout << "-------- Test of " << __func__ << ": Fail! -------- " << std::endl;
    else
        std::cout << "-------- Test of " << __func__ << ": Passed! -------- " << std::endl;
}

void test_BMM_S8T_S8N_F32T() {
    const int b = 12, m = 512, k = 64, n = 512;
    const float alpha = 0.0006456375122070312;
    MemoryAllocator mem_buf;

    int8_t *intput_arr = mem_buf.get_int8buffer(b * m * k);
    int8_t *weight_arr = mem_buf.get_int8buffer(b * k * n);
    float *output_arr = mem_buf.get_fpbuffer(b * m * n);
    float *GToutput_arr = mem_buf.get_fpbuffer(b * m * n);

    Matrix3D<int8_t> input(intput_arr, b, m, k);
    Matrix3D<int8_t> weight(weight_arr, b, n, k);
    Matrix3D<float> output(output_arr, b, m, n);
    Matrix3D<float> GToutput(GToutput_arr, b, m, n);

    input.load("assets/OPT/tests/ops/OPT_125m/BMM_S8T_S8N_F32T_x.bin");
    weight.load("assets/OPT/tests/ops/OPT_125m/BMM_S8T_S8N_F32T_weight.bin");
    GToutput.load("assets/OPT/tests/ops/OPT_125m/BMM_S8T_S8N_F32T_y.bin");

    struct BMM_S8T_S8N_F32T_params op_params = {alpha};

    auto test_op = BMM_S8T_S8N_F32T(op_params);
    test_op.forward(input, weight, output);

    bool success = check_two_equal(output_arr, GToutput_arr, b * m * n);

    if (!success)
        std::cout << "-------- Test of " << __func__ << ": Fail! -------- " << std::endl;
    else
        std::cout << "-------- Test of " << __func__ << ": Passed! -------- " << std::endl;
}

void test_BMM_S8T_S8N_F32T_1_3B() {
    const int heads = 32, sqlen = 108, c = 64;
    MemoryAllocator mem_buf;

    int8_t *intput_arr = mem_buf.get_int8buffer(heads * sqlen * c);
    int8_t *weight_arr = mem_buf.get_int8buffer(heads * sqlen * c);
    float *output_arr = mem_buf.get_fpbuffer(heads * sqlen * sqlen);
    float *GToutput_arr = mem_buf.get_fpbuffer(heads * sqlen * sqlen);

    Matrix3D<int8_t> input(intput_arr, heads, sqlen, c);
    Matrix3D<int8_t> weight(weight_arr, heads, sqlen, c);
    Matrix3D<float> output(output_arr, heads, sqlen, sqlen);
    Matrix3D<float> GToutput(GToutput_arr, heads, sqlen, sqlen);

    input.load("assets/OPT/tests/ops/OPT_1.3B/BMM_S8T_S8N_F32T_x1.bin");
    weight.load("assets/OPT/tests/ops/OPT_1.3B/BMM_S8T_S8N_F32T_x2.bin");
    GToutput.load("assets/OPT/tests/ops/OPT_1.3B/BMM_S8T_S8N_F32T_y.bin");

    struct BMM_S8T_S8N_F32T_params op_params = {};

    auto test_op = BMM_S8T_S8N_F32T(op_params);
    load_BMM_S8T_S8N_F32T(test_op, "models/OPT_1.3B/decoder/layer0/self_attn/qk_bmm/");
    test_op.forward(input, weight, output);

    bool success = check_two_equal(output_arr, GToutput_arr, GToutput.length());

    if (!success)
        std::cout << "-------- Test of " << __func__ << ": Fail! -------- " << std::endl;
    else
        std::cout << "-------- Test of " << __func__ << ": Passed! -------- " << std::endl;
}

void test_BMM_S8T_S8N_S8T() {
    const int b = 12, m = 512, k = 512, n = 64;
    const float alpha = 0.013031005859375;
    MemoryAllocator mem_buf;

    int8_t *intput_arr = mem_buf.get_int8buffer(b * m * k);
    int8_t *weight_arr = mem_buf.get_int8buffer(b * k * n);
    int8_t *output_arr = mem_buf.get_int8buffer(b * m * n);
    int8_t *GToutput_arr = mem_buf.get_int8buffer(b * m * n);

    Matrix3D<int8_t> input(intput_arr, b, m, k);
    Matrix3D<int8_t> weight(weight_arr, b, n, k);
    Matrix3D<int8_t> output(output_arr, b, m, n);
    Matrix3D<int8_t> GToutput(GToutput_arr, b, m, n);

    input.load("assets/OPT/tests/ops/OPT_125m/BMM_S8T_S8N_S8T_x.bin");
    weight.load("assets/OPT/tests/ops/OPT_125m/BMM_S8T_S8N_S8T_weight.bin");
    GToutput.load("assets/OPT/tests/ops/OPT_125m/BMM_S8T_S8N_S8T_y.bin");

    struct BMM_S8T_S8N_S8T_params op_params = {alpha};

    auto test_op = BMM_S8T_S8N_S8T(op_params);
    test_op.forward(input, weight, output);

    bool success = check_two_exact_equal(output_arr, GToutput_arr, GToutput.length());
    if (!success)
        std::cout << "-------- Test of " << __func__ << ": Fail! -------- " << std::endl;
    else
        std::cout << "-------- Test of " << __func__ << ": Passed! -------- " << std::endl;
}

void test_BMM_S8T_S8N_S8T_1_3B() {
    const int heads = 32, sqlen = 108, c = 64;
    const float alpha = 0.00787353515625;
    MemoryAllocator mem_buf;

    int8_t *intput_arr = mem_buf.get_int8buffer(heads * sqlen * sqlen);
    int8_t *weight_arr = mem_buf.get_int8buffer(heads * c * sqlen);
    int8_t *output_arr = mem_buf.get_int8buffer(heads * sqlen * c);
    int8_t *GToutput_arr = mem_buf.get_int8buffer(heads * sqlen * c);

    Matrix3D<int8_t> input(intput_arr, heads, sqlen, sqlen);
    Matrix3D<int8_t> weight(weight_arr, heads, c, sqlen);
    Matrix3D<int8_t> output(output_arr, heads, sqlen, c);
    Matrix3D<int8_t> GToutput(GToutput_arr, heads, sqlen, c);

    input.load("assets/OPT/tests/ops/OPT_1.3B/BMM_S8T_S8N_S8T_x1.bin");
    weight.load("assets/OPT/tests/ops/OPT_1.3B/BMM_S8T_S8N_S8T_x2.bin");
    GToutput.load("assets/OPT/tests/ops/OPT_1.3B/BMM_S8T_S8N_S8T_y.bin");

    struct BMM_S8T_S8N_S8T_params op_params = {alpha};

    auto test_op = BMM_S8T_S8N_S8T(op_params);
    test_op.forward(input, weight, output);

    bool success = check_two_exact_equal(output_arr, GToutput_arr, GToutput.length());
    if (!success)
        std::cout << "-------- Test of " << __func__ << ": Fail! -------- " << std::endl;
    else
        std::cout << "-------- Test of " << __func__ << ": Passed! -------- " << std::endl;
}

void test_Embedding() {
    const int voc_size = 50272, embed_dim = 768, sqlen = 512, padding_idx = 1;
    MemoryAllocator mem_buf;

    Matrix3D<int> input(mem_buf.get_intbuffer(sqlen), 1, 1, sqlen);
    Matrix3D<float> weight(mem_buf.get_fpbuffer(voc_size * embed_dim), 1, voc_size, embed_dim);
    Matrix3D<float> output(mem_buf.get_fpbuffer(sqlen * embed_dim), 1, sqlen, embed_dim);
    Matrix3D<float> outputGT(mem_buf.get_fpbuffer(sqlen * embed_dim), 1, sqlen, embed_dim);

    input.load("assets/OPT/tests/ops/input_ids.bin");
    outputGT.load("assets/OPT/tests/ops/OPT_125m/inputs_embeds.bin");

    auto embed_tokens = Embedding(embed_dim, voc_size, padding_idx, weight);
    load_Embedding_params(embed_tokens, "models/OPT_125m/decoder/embed_tokens");

    embed_tokens.forward(input, output);
    bool success = check_two_equal(output.m_data, outputGT.m_data, sqlen * embed_dim);

    if (!success)
        std::cout << "-------- Test of " << __func__ << ": Fail! -------- " << std::endl;
    else
        std::cout << "-------- Test of " << __func__ << ": Passed! -------- " << std::endl;
}

void test_Embedding_1_3B() {
    const int voc_size = 50272, embed_dim = 2048, sqlen = 108, padding_idx = 1;
    MemoryAllocator mem_buf;

    Matrix3D<int> input(mem_buf.get_intbuffer(sqlen), 1, 1, sqlen);
    Matrix3D<float> weight(mem_buf.get_fpbuffer(voc_size * embed_dim), 1, voc_size, embed_dim);
    Matrix3D<float> output(mem_buf.get_fpbuffer(sqlen * embed_dim), 1, sqlen, embed_dim);
    Matrix3D<float> outputGT(mem_buf.get_fpbuffer(sqlen * embed_dim), 1, sqlen, embed_dim);

    input.load("assets/OPT/tests/ops/input_ids.bin");
    outputGT.load("assets/OPT/tests/ops/OPT_1.3B/inputs_embeds.bin");

    auto embed_tokens = Embedding(embed_dim, voc_size, padding_idx, weight);
    load_Embedding_params(embed_tokens, "models/OPT_1.3B/decoder/embed_tokens");

    embed_tokens.forward(input, output);
    bool success = check_two_equal(output.m_data, outputGT.m_data, sqlen * embed_dim);

    if (!success)
        std::cout << "-------- Test of " << __func__ << ": Fail! -------- " << std::endl;
    else
        std::cout << "-------- Test of " << __func__ << ": Passed! -------- " << std::endl;
}

void test_LlamaRMSNorm() {
    const struct model_config llama7B = llama_7B;
    const int sqlen = 9, embed_dim = llama7B.embed_dim;

    MemoryAllocator mem_buf;

    Matrix3D<float> hidden_states(mem_buf.get_fpbuffer(sqlen * embed_dim), 1, sqlen, embed_dim);
    Matrix3D<float> weight(mem_buf.get_fpbuffer(embed_dim), 1, 1, embed_dim);
    Matrix3D<float> outputGT(mem_buf.get_fpbuffer(sqlen * embed_dim), 1, sqlen, embed_dim);
    Matrix3D<float> output(mem_buf.get_fpbuffer(sqlen * embed_dim), 1, sqlen, embed_dim);

    hidden_states.load("assets/llama/tests/ops/RMSnorm/hidden_states.bin");
    weight.load("assets/llama/tests/ops/RMSnorm/weight.bin");
    outputGT.load("assets/llama/tests/ops/RMSnorm/output.bin");

    LlamaRMSNorm op(weight);

    op.forward(hidden_states, output);
    bool success = check_two_equal(output.m_data, outputGT.m_data, sqlen * embed_dim);

    if (!success)
        std::cout << "-------- Test of " << __func__ << ": Fail! -------- " << std::endl;
    else
        std::cout << "-------- Test of " << __func__ << ": Passed! -------- " << std::endl;
}

// TODO: test fp32/fp32, fp16/fp16, fp32/w4, fp16/w4
void test_FPLinear() {
    const int m = 1, n = 32000, k = 4096;

    MemoryAllocator mem_buf;

    Matrix3D<float> hidden_states(mem_buf.get_fpbuffer(m * k), 1, m, k);
    Matrix3D<float> weight(mem_buf.get_fpbuffer(n * k), 1, n, k);
    Matrix3D<float> outputGT(mem_buf.get_fpbuffer(m * n), 1, m, n);
    Matrix3D<float> output(mem_buf.get_fpbuffer(m * n), 1, m, n);

    hidden_states.load("assets/llama/tests/ops/Linear/input.bin");
    outputGT.load("assets/llama/tests/ops/Linear/output.bin");
    Linear_FP op(weight, "models/LLaMA_7B/lm_head.bin");

    op.forward(hidden_states, output);

    bool success = check_two_equal(output.m_data, outputGT.m_data, output.length(), 1e-8);

    if (!success)
        std::cout << "-------- Test of " << __func__ << ": Fail! -------- " << std::endl;
    else
        std::cout << "-------- Test of " << __func__ << ": Passed! -------- " << std::endl;
}

// TODO: test fp32/fp32, fp16/fp16, fp32/w4, fp16/w4
void test_FP16Linear_int4() {
    const int m = 1, n = 32000, k = 4096;

    float *hidden_states_arr;
    allocate_aligned_memory(hidden_states_arr, (m * k * sizeof(float)));
    Matrix3D<float> hidden_states(hidden_states_arr, 1, m, k);
    hidden_states.load("assets/llama/tests/ops/Linear/input.bin");

    float16_t *hidden_states_ref_arr;
    allocate_aligned_memory_gpu(hidden_states_ref_arr, (m * k * sizeof(float16_t)));
    Matrix3D<float16_t> hidden_states_ref(hidden_states_ref_arr, 1, m, k);

    half *hidden_states_cuda_arr;
    allocate_aligned_memory_gpu(hidden_states_cuda_arr, (m * k * sizeof(half)));
    Matrix3D<half> hidden_states_cuda(hidden_states_cuda_arr, 1, m, k);

    for(int i = 0; i < m * k; i++) {
        hidden_states_ref_arr[i] = static_cast<float16_t>(hidden_states_arr[i]);
        hidden_states_cuda_arr[i] = __float2half(hidden_states_arr[i]);
    }

    int32_t *int4_weight_ref_arr;
    allocate_aligned_memory(int4_weight_ref_arr, (n * k / 8 * sizeof(int32_t)));
    Matrix3D<int32_t> int4_ref_weight(int4_weight_ref_arr, 1, n / 8, k);
    Linear_FP16_int4_ref int4_op_ref = Linear_FP16_int4_ref(int4_ref_weight, "models/LLaMA_7B/lm_head/");

    int32_t *int4_weight_cuda_arr;
    allocate_aligned_memory_gpu(int4_weight_cuda_arr, (n * k / 8 * sizeof(int32_t)));
    Matrix3D<int32_t> int4_cuda_weight(int4_weight_cuda_arr, 1, n / 8, k);
    Linear_half_int4_test int4_op_cuda = Linear_half_int4_test(int4_cuda_weight, "models/LLaMA_7B/lm_head/");

    float16_t *outputQ_ref_arr;
    allocate_aligned_memory_gpu(outputQ_ref_arr, (m * n * sizeof(float16_t)));
    Matrix3D<float16_t> outputQ_ref(outputQ_ref_arr, 1, m, n);
    
    half *outputQ_cuda_arr;
    allocate_aligned_memory_gpu(outputQ_cuda_arr, (m * n * sizeof(half)));
    Matrix3D<half> outputQ_cuda(outputQ_cuda_arr, 1, m, n);

    const int flops = k * m * n * 2;
    STATS_FLOPS("int4_ref", flops);
    int4_op_ref.forward_ref(hidden_states_ref, outputQ_ref);
    STATS_END("int4_ref");
    STATS_FLOPS("int4_fast", flops);
    int4_op_cuda.forward(hidden_states_cuda, outputQ_cuda);
    cudaDeviceSynchronize();
    STATS_END("int4_fast");

    bool success = check_two_equal_cpu_gpu(outputQ_ref.m_data, outputQ_cuda.m_data, outputQ_ref.length(), 7e-4);

    if (!success)
        std::cout << "-------- Test of " << __func__ << ": Fail! -------- " << std::endl;
    else
        std::cout << "-------- Test of " << __func__ << ": Passed! -------- " << std::endl;
}

// TODO: test fp32/fp32, fp16/fp16, fp32/w4, fp16/w4
void test_FP16Linear_int4_mini() {
    const int m = 1, n = 64, k = 32;

    float16_t *hidden_states_ref_arr;
    allocate_aligned_memory_gpu(hidden_states_ref_arr, (m * k * sizeof(float16_t)));
    Matrix3D<float16_t> hidden_states_ref(hidden_states_ref_arr, 1, m, k);

    half *hidden_states_cuda_arr;
    allocate_aligned_memory_gpu(hidden_states_cuda_arr, (m * k * sizeof(half)));
    Matrix3D<half> hidden_states_cuda(hidden_states_cuda_arr, 1, m, k);

    printf("a\n");

    for(int i = 0; i < m * k; i++) {
        float v = float(i % 4) / 4;
        hidden_states_ref_arr[i] = static_cast<float16_t>(v);
        hidden_states_cuda_arr[i] = __float2half(v);
    }

    printf("b\n");

    int32_t *int4_weight_ref_arr;
    allocate_aligned_memory(int4_weight_ref_arr, (n * k / 8 * sizeof(int32_t)));
    Matrix3D<int32_t> int4_ref_weight(int4_weight_ref_arr, 1, n / 8, k);
    Linear_FP16_int4_ref int4_op_ref;

    float16_t *scale_ref_arr;
    allocate_aligned_memory_gpu(scale_ref_arr, (n * k / 32 * sizeof(float16_t)));
    Matrix3D<float16_t> int4_ref_scale(scale_ref_arr, 1, n / 32, k);

    int *zero_ref_arr;
    allocate_aligned_memory(zero_ref_arr, (n * k / 32 * sizeof(half)));
    Matrix3D<int> int4_ref_zero(zero_ref_arr, 1, n / 32, k);
    
    int4_op_ref.weight = int4_ref_weight;
    int4_op_ref.scale = int4_ref_scale;
    int4_op_ref.zero_point = int4_ref_zero;

    int32_t *int4_weight_cuda_arr;
    allocate_aligned_memory_gpu(int4_weight_cuda_arr, (n * k / 8 * sizeof(int32_t)));
    Matrix3D<int32_t> int4_cuda_weight(int4_weight_cuda_arr, 1, n / 8, k);
    Linear_half_int4_test int4_op_cuda;

    half *scale_cuda_arr;
    allocate_aligned_memory_gpu(scale_cuda_arr, (n * k / 32 * sizeof(half)));
    Matrix3D<half> int4_cuda_scale(scale_cuda_arr, 1, n / 32, k);

    int *zero_cuda_arr;
    allocate_aligned_memory_gpu(zero_cuda_arr, (n * k / 32 * sizeof(int)));
    Matrix3D<int> int4_cuda_zero(zero_cuda_arr, 1, n / 32, k);

    int4_op_cuda.weight = int4_cuda_weight;
    int4_op_cuda.scale = int4_cuda_scale;
    int4_op_cuda.zero_point = int4_cuda_zero;

    printf("c\n");

    for(int i = 0; i < n * k / 8; i++) {
        const int const_w = 0x23413221;
        int4_weight_ref_arr[i] = const_w;
        int4_weight_cuda_arr[i] = const_w;
    }
    printf("d\n");
    for(int i = 0; i < n * k / 32; i++) {
        float v = float(i % 4) / 4;
        scale_ref_arr[i] = static_cast<float16_t>(v);
        scale_cuda_arr[i] = __float2half(v);
    }
    printf("e\n");
    for(int i = 0; i < n * k / 32; i++) {
        const int const_z = 0x88888888;
        zero_ref_arr[i] = const_z;
        zero_cuda_arr[i] = const_z;
    }
    printf("f\n");

    cudaDeviceSynchronize();

    float16_t *outputQ_ref_arr;
    allocate_aligned_memory_gpu(outputQ_ref_arr, (m * n * sizeof(float16_t)));
    Matrix3D<float16_t> outputQ_ref(outputQ_ref_arr, 1, m, n);
    half *outputQ_cuda_arr;
    allocate_aligned_memory_gpu(outputQ_cuda_arr, (m * n * sizeof(half)));
    Matrix3D<half> outputQ_cuda(outputQ_cuda_arr, 1, m, n);

    printf("g\n");

    const int flops = k * m * n * 2;
    STATS_FLOPS("int4_ref", flops);
    int4_op_ref.forward_ref(hidden_states_ref, outputQ_ref);
    STATS_END("int4_ref");
    STATS_FLOPS("int4_fast", flops);
    int4_op_cuda.forward(hidden_states_cuda, outputQ_cuda);
    cudaDeviceSynchronize();
    STATS_END("int4_fast");

    printf("h\n");

    bool success = check_two_equal_cpu_gpu(outputQ_ref.m_data, outputQ_cuda.m_data, outputQ_ref.length(), 1e-10);

    printf("i\n");

    if (!success)
        std::cout << "-------- Test of " << __func__ << ": Fail! -------- " << std::endl;
    else
        std::cout << "-------- Test of " << __func__ << ": Passed! -------- " << std::endl;
}

int main() {
    test_LlamaRMSNorm_cuda();
    test_W8A8B8O8LinearReLU();
    test_W8A8B8O8LinearReLU_1_3B();
    test_W8A8B8O8Linear();
    test_W8A8B8O8Linear_1_3B();
    test_W8A8BFP32OFP32Linear();
    test_W8A8BFP32OFP32Linear_1_3B();
    test_BMM_S8T_S8N_F32T();
    test_BMM_S8T_S8N_F32T_1_3B();
    test_BMM_S8T_S8N_S8T();
    test_BMM_S8T_S8N_S8T_1_3B();
    test_Embedding();
    test_Embedding_1_3B();
    // LLaMa
    test_LlamaRMSNorm();
    test_FPLinear();
    test_FP16Linear_int4();
    // test_FP16Linear_int4_mini();
    // Report if profiling flag is on
    Profiler::getInstance().report();
}
