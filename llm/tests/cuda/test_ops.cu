#include <cmath>

#include "operators.h"
#include "utils.h"
#include "../utils_memalloc.h"

void test_LayerNormQ() {
    const int b = 1, m = 108, n = 768;
    MemoryAllocator mem_buf;

    float *intput_arr = mem_buf.get_fpbuffer(b * m * n);
    float *weight_arr = mem_buf.get_fpbuffer(b * n);
    float *bias_arr = mem_buf.get_fpbuffer(b * n);
    int8_t *output_arr = mem_buf.get_int8buffer(b * m * n);
    int8_t *GToutput_arr = mem_buf.get_int8buffer(b * m * n);

    Matrix3D<float> input(intput_arr, b, m, n);
    Matrix3D<float> weight(weight_arr, b, 1, n);
    Matrix3D<float> bias(bias_arr, b, 1, n);
    Matrix3D<int8_t> output(output_arr, b, m, n);
    Matrix3D<int8_t> GToutput(GToutput_arr, b, m, n);

    bias.load("assets/OPT/tests/ops/OPT_125m/LayerNormQ_bias.bin");
    input.load("assets/OPT/tests/ops/OPT_125m/LayerNormQ_x.bin");
    weight.load("assets/OPT/tests/ops/OPT_125m/LayerNormQ_weight.bin");
    GToutput.load("assets/OPT/tests/ops/OPT_125m/LayerNormQ_out.bin");

    struct LayerNormQ_params op_params = {weight, bias};

    LayerNormQ test_op = LayerNormQ(op_params);

    test_op.forward(input, output);

    bool success = check_two_exact_equal(output_arr, GToutput_arr, b * m * n);
    if (!success)
        std::cout << "-------- Test of " << __func__ << ": Fail! -------- " << std::endl;
    else
        std::cout << "-------- Test of " << __func__ << ": Passed! -------- " << std::endl;
}

void test_LayerNormQ_len512() {
    const int b = 1, m = 512, n = 768;
    MemoryAllocator mem_buf;

    float *intput_arr = mem_buf.get_fpbuffer(b * m * n);
    float *weight_arr = mem_buf.get_fpbuffer(b * n);
    float *bias_arr = mem_buf.get_fpbuffer(b * n);
    int8_t *output_arr = mem_buf.get_int8buffer(b * m * n);
    int8_t *GToutput_arr = mem_buf.get_int8buffer(b * m * n);

    Matrix3D<float> input(intput_arr, b, m, n);
    Matrix3D<float> weight(weight_arr, b, 1, n);
    Matrix3D<float> bias(bias_arr, b, 1, n);
    Matrix3D<int8_t> output(output_arr, b, m, n);
    Matrix3D<int8_t> GToutput(GToutput_arr, b, m, n);

    input.load("assets/OPT/tests/ops/OPT_125m/LayerNormQ_x_len512.bin");
    GToutput.load("assets/OPT/tests/ops/OPT_125m/LayerNormQ_y_len512.bin");

    struct LayerNormQ_params op_params = {weight, bias};

    LayerNormQ test_op = LayerNormQ(op_params);
    load_LayerNormQ(test_op, "models/OPT_125m/decoder/layer0/self_attn_layer_norm");

    test_op.forward(input, output);

    bool success = check_two_equal(output_arr, GToutput_arr, b * m * n);
    if (!success)
        std::cout << "-------- Test of " << __func__ << ": Fail! -------- " << std::endl;
    else
        std::cout << "-------- Test of " << __func__ << ": Passed! -------- " << std::endl;
}

void test_LayerNormQ_1_3B() {
    const int b = 1, m = 108, n = 2048;
    MemoryAllocator mem_buf;

    float *intput_arr = mem_buf.get_fpbuffer(b * m * n);
    float *weight_arr = mem_buf.get_fpbuffer(b * n);
    float *bias_arr = mem_buf.get_fpbuffer(b * n);
    int8_t *output_arr = mem_buf.get_int8buffer(b * m * n);
    int8_t *GToutput_arr = mem_buf.get_int8buffer(b * m * n);

    Matrix3D<float> input(intput_arr, b, m, n);
    Matrix3D<float> weight(weight_arr, b, 1, n);
    Matrix3D<float> bias(bias_arr, b, 1, n);
    Matrix3D<int8_t> output(output_arr, b, m, n);
    Matrix3D<int8_t> GToutput(GToutput_arr, b, m, n);

    input.load("assets/OPT/tests/ops/OPT_1.3B/LayerNormQ_x.bin");
    GToutput.load("assets/OPT/tests/ops/OPT_1.3B/LayerNormQ_out.bin");

    struct LayerNormQ_params op_params = {weight, bias};

    LayerNormQ op = LayerNormQ(op_params);
    load_LayerNormQ(op, "models/OPT_1.3B/decoder/layer0/self_attn_layer_norm/");

    LayerNormQ test_op = LayerNormQ(op_params);

    test_op.forward(input, output);

    bool success = check_two_exact_equal(output_arr, GToutput_arr, b * m * n);
    if (!success)
        std::cout << "-------- Test of " << __func__ << ": Fail! -------- " << std::endl;
    else
        std::cout << "-------- Test of " << __func__ << ": Passed! -------- " << std::endl;
}

void test_LayerNorm() {
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

void test_LayerNorm_1_3B_len512() {
    const int b = 1, m = 512, n = 2048;
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

    input.load("assets/OPT/tests/ops/OPT_1.3B/decoder/final_layer_norm_hidden_states.bin");
    GToutput.load("assets/OPT/tests/ops/OPT_1.3B/decoder/final_layer_norm_output.bin");

    struct LayerNorm_params op_params = {weight, bias};

    LayerNorm test_op = LayerNorm(op_params);
    load_LayerNorm(test_op, "models/OPT_1.3B/decoder/final_layer_norm/");

    test_op.forward(input, output);

    bool success = check_two_equal(output_arr, GToutput_arr, b * m * n, 8e-6);
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
    Linear_FP op(weight, "models/LLaMA_7B_2_chat/lm_head.bin");

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

    naive_float16_t *hidden_states_ref_arr;
    allocate_aligned_memory_gpu(hidden_states_ref_arr, (m * k * sizeof(naive_float16_t)));
    Matrix3D<naive_float16_t> hidden_states_ref(hidden_states_ref_arr, 1, m, k);

    half *hidden_states_cuda_arr;
    allocate_aligned_memory_gpu(hidden_states_cuda_arr, (m * k * sizeof(half)));
    Matrix3D<half> hidden_states_cuda(hidden_states_cuda_arr, 1, m, k);

    for(int i = 0; i < m * k; i++) {
        hidden_states_ref_arr[i] = static_cast<naive_float16_t>(hidden_states_arr[i]);
        hidden_states_cuda_arr[i] = __float2half(hidden_states_arr[i]);
    }

    int32_t *int4_weight_ref_arr;
    allocate_aligned_memory(int4_weight_ref_arr, (n * k / 8 * sizeof(int32_t)));
    Matrix3D<int32_t> int4_ref_weight(int4_weight_ref_arr, 1, n / 8, k);
    Linear_FP16_int4_ref int4_op_ref = Linear_FP16_int4_ref(int4_ref_weight, "INT4/models/LLaMA_7B_2_chat/lm_head/");

    int32_t *int4_weight_cuda_arr;
    allocate_aligned_memory_gpu(int4_weight_cuda_arr, (n * k / 8 * sizeof(int32_t)));
    Matrix3D<int32_t> int4_cuda_weight(int4_weight_cuda_arr, 1, n / 8, k);
    Linear_half_int4 int4_op_cuda = Linear_half_int4(int4_cuda_weight, "INT4/models/LLaMA_7B_2_chat/lm_head/");

    naive_float16_t *outputQ_ref_arr;
    allocate_aligned_memory_gpu(outputQ_ref_arr, (m * n * sizeof(naive_float16_t)));
    Matrix3D<naive_float16_t> outputQ_ref(outputQ_ref_arr, 1, m, n);

    half *outputQ_cuda_arr;
    allocate_aligned_memory_gpu(outputQ_cuda_arr, (m * n * sizeof(half)));
    Matrix3D<half> outputQ_cuda(outputQ_cuda_arr, 1, m, n);

    half *split_k_buffer;
    allocate_aligned_memory_gpu(split_k_buffer, (m * n * 8 * sizeof(half)));

    const int flops = k * m * n * 2;
    STATS_FLOPS("int4_ref", flops);
    int4_op_ref.forward_ref(hidden_states_ref, outputQ_ref);
    STATS_END("int4_ref");
    STATS_FLOPS("int4_fast", flops);
    int4_op_cuda.forward(hidden_states_cuda, outputQ_cuda, split_k_buffer);
    cudaDeviceSynchronize();
    STATS_END("int4_fast");

    bool success = check_two_equal_cpu_gpu(outputQ_ref.m_data, outputQ_cuda.m_data, outputQ_ref.length(), 7e-4);

    if (!success)
        std::cout << "-------- Test of " << __func__ << ": Fail! -------- " << std::endl;
    else
        std::cout << "-------- Test of " << __func__ << ": Passed! -------- " << std::endl;

    // Free memory
    cudaFree(hidden_states_ref_arr);
    cudaFree(hidden_states_cuda_arr);
    cudaFree(int4_weight_cuda_arr);
    cudaFree(outputQ_ref_arr);
    cudaFree(outputQ_cuda_arr);
    cudaFree(split_k_buffer);
}

void test_LlamaRMSNorm_cuda() {
    const struct model_config llama7B = llama_7B;
    const int sqlen = 9, embed_dim = llama7B.embed_dim;

    half* buffer_1;
    cudaMallocManaged(&buffer_1, sizeof(half) * sqlen * embed_dim);
    Matrix3D<half> hidden_states(buffer_1, 1, sqlen, embed_dim);
    read_to_array_half("assets/llama/tests/ops/RMSnorm/hidden_states_half.bin", hidden_states.m_data, sqlen * embed_dim);

    float* buffer_2;
    cudaMallocManaged(&buffer_2, sizeof(float) * embed_dim);
    Matrix3D<float> weight(buffer_2, 1, 1, embed_dim);
    read_to_array("assets/llama/tests/ops/RMSnorm/weight.bin", weight.m_data, embed_dim);

    half* buffer_3;
    cudaMallocManaged(&buffer_3, sizeof(half) * sqlen * embed_dim);
    Matrix3D<half> outputGT(buffer_3, 1, sqlen, embed_dim);
    read_to_array_half("assets/llama/tests/ops/RMSnorm/output_half.bin", outputGT.m_data, sqlen * embed_dim);

    half* buffer_4;
    cudaMallocManaged(&buffer_4, sizeof(half) * sqlen * embed_dim);
    Matrix3D<half> output(buffer_4, 1, sqlen, embed_dim);

    LlamaRMSNorm_cuda op(weight);
    op.forward(hidden_states, output);
    cudaDeviceSynchronize();

    bool success = check_two_equal_half_half(output.m_data, outputGT.m_data, sqlen * embed_dim);

    if (!success)
        std::cout << "-------- Test of " << __func__ << ": Fail! -------- " << std::endl;
    else
        std::cout << "-------- Test of " << __func__ << ": Passed! -------- " << std::endl;

    cudaFree(buffer_1);
    cudaFree(buffer_2);
    cudaFree(buffer_3);
    cudaFree(buffer_4);
}

void test_softmax_cuda() {
    const struct model_config llama7B = llama_7B;
    const int sqlen = 9, past_sqlen = 0, num_heads = llama7B.num_heads;
    const int tgz = (sqlen + past_sqlen);

    half* buffer_1;
    cudaMallocManaged(&buffer_1, sizeof(half) * num_heads * sqlen * tgz);
    Matrix3D<half> attn_weights(buffer_1, num_heads, sqlen, tgz);
    read_to_array_half("assets/llama/tests/ops/softmax/input_half.bin", attn_weights.m_data, num_heads * sqlen * tgz);

    half* buffer_2;
    cudaMallocManaged(&buffer_2, sizeof(half) * num_heads * sqlen * tgz);
    Matrix3D<half> attn_probsGT(buffer_2, num_heads, sqlen, tgz);
    read_to_array_half("assets/llama/tests/ops/softmax/output_half.bin", attn_probsGT.m_data, num_heads * sqlen * tgz);

    half* buffer_3;
    cudaMallocManaged(&buffer_3, sizeof(half) * num_heads * sqlen * tgz);
    Matrix3D<half> attn_probs(buffer_3, num_heads, sqlen, tgz);

    int blockSize = 32;
    dim3 threadsPerBlock(blockSize, blockSize);
    dim3 numBlocksPerGrid((num_heads + blockSize - 1) / blockSize, (sqlen + blockSize - 1) / blockSize);
    softmax_cuda<<<numBlocksPerGrid, threadsPerBlock>>>(attn_weights, attn_probs);
    cudaDeviceSynchronize();

    bool success = check_two_equal_half_half(attn_probs.m_data, attn_probsGT.m_data, num_heads * sqlen * tgz);

    if (!success)
        std::cout << "-------- Test of " << __func__ << ": Fail! -------- " << std::endl;
    else
        std::cout << "-------- Test of " << __func__ << ": Passed! -------- " << std::endl;

    cudaFree(buffer_1);
    cudaFree(buffer_2);
    cudaFree(buffer_3);
}

void test_BMM_F16T() {
    const struct model_config llama7B = llama_7B;
    const int sqlen = 9, past_sqlen = 0, embed_dim = llama7B.embed_dim, num_heads = llama7B.num_heads,
              head_dim = embed_dim / num_heads;
    const int tgz = (sqlen + past_sqlen);
    const half alpha = __float2half(0.088388);

    half* buffer_1;
    cudaMallocManaged(&buffer_1, sizeof(half) * num_heads * sqlen * head_dim);
    Matrix3D<half> query_states(buffer_1, num_heads, sqlen, head_dim);
    read_to_array_half("assets/llama/tests/ops/BMM_F16T/input_half.bin", query_states.m_data, num_heads * sqlen * head_dim);

    half* buffer_2;
    cudaMallocManaged(&buffer_2, sizeof(half) * num_heads * tgz * head_dim);
    Matrix3D<half> final_key_states(buffer_2, num_heads, tgz, head_dim);
    read_to_array_half("assets/llama/tests/ops/BMM_F16T/weight_half.bin", final_key_states.m_data, num_heads * tgz * head_dim);

    half* buffer_3;
    cudaMallocManaged(&buffer_3, sizeof(half) * num_heads * sqlen * tgz);
    Matrix3D<half> attn_weightsGT(buffer_3, num_heads, sqlen, tgz);
    read_to_array_half("assets/llama/tests/ops/BMM_F16T/output_half.bin", attn_weightsGT.m_data, num_heads * sqlen * tgz);

    half* buffer_4;
    cudaMallocManaged(&buffer_4, sizeof(half) * num_heads * sqlen * tgz);
    Matrix3D<half> attn_weights(buffer_4, num_heads, sqlen, tgz);

    BMM_F16T qk_bmm = BMM_F16T(alpha);
    qk_bmm.forward(query_states, final_key_states, attn_weights);
    cudaDeviceSynchronize();

    bool success = check_two_equal_half_half(attn_weights.m_data, attn_weightsGT.m_data, num_heads * sqlen * tgz);

    if (!success)
        std::cout << "-------- Test of " << __func__ << ": Fail! -------- " << std::endl;
    else
        std::cout << "-------- Test of " << __func__ << ": Passed! -------- " << std::endl;

    cudaFree(buffer_1);
    cudaFree(buffer_2);
    cudaFree(buffer_3);
    cudaFree(buffer_4);
}

void test_RotaryPosEmb_cuda() {
    const struct model_config llama7B = llama_7B;
    const int sqlen = 9, embed_dim = llama7B.embed_dim, num_heads = llama7B.num_heads, head_dim = embed_dim / num_heads;
    const int max_sqlen = 2048;
    const int start_idx = 0;

    half* buffer_1;
    cudaMallocManaged(&buffer_1, sizeof(half) * num_heads * sqlen * head_dim);
    Matrix3D<half> query_states(buffer_1, num_heads, sqlen, head_dim);
    read_to_array_half("assets/llama/tests/ops/RotaryPosEmb/input_query_half.bin", query_states.m_data, num_heads * sqlen * head_dim);

    half* buffer_2;
    cudaMallocManaged(&buffer_2, sizeof(half) * num_heads * sqlen * head_dim);
    Matrix3D<half> key_states(buffer_2, num_heads, sqlen, head_dim);
    read_to_array_half("assets/llama/tests/ops/RotaryPosEmb/input_key_half.bin", key_states.m_data, num_heads * sqlen * head_dim);

    half* buffer_3;
    cudaMallocManaged(&buffer_3, sizeof(half) * max_sqlen * (embed_dim / num_heads));
    Matrix3D<half> cos(buffer_3, 1, max_sqlen, (embed_dim / num_heads));
    read_to_array_half("assets/llama/tests/ops/RotaryPosEmb/cos_cached_half.bin", cos.m_data, max_sqlen * (embed_dim / num_heads));

    half* buffer_4;
    cudaMallocManaged(&buffer_4, sizeof(half) * max_sqlen * (embed_dim / num_heads));
    Matrix3D<half> sin(buffer_4, 1, max_sqlen, (embed_dim / num_heads));
    read_to_array_half("assets/llama/tests/ops/RotaryPosEmb/sin_cached_half.bin", sin.m_data, max_sqlen * (embed_dim / num_heads));

    half* buffer_5;
    cudaMallocManaged(&buffer_5, sizeof(half) * num_heads * sqlen * head_dim);
    Matrix3D<half> query_statesGT(buffer_5, num_heads, sqlen, head_dim);
    read_to_array_half("assets/llama/tests/ops/RotaryPosEmb/output_query_half.bin", query_statesGT.m_data, num_heads * sqlen * head_dim);

    half* buffer_6;
    cudaMallocManaged(&buffer_6, sizeof(half) * num_heads * sqlen * head_dim);
    Matrix3D<half> key_statesGT(buffer_6, num_heads, sqlen, head_dim);
    read_to_array_half("assets/llama/tests/ops/RotaryPosEmb/output_key_half.bin", key_statesGT.m_data, num_heads * sqlen * head_dim);

    dim3 grid(num_heads, 1, 1);
    dim3 block(sqlen, 1, 1);
    RotaryPosEmb_cuda_forward<<<grid, block>>>(query_states, key_states, cos, sin, start_idx, sqlen);
    cudaDeviceSynchronize();

    bool success = check_two_equal_half_half(query_states.m_data, query_statesGT.m_data, num_heads * sqlen * head_dim);
    success &= check_two_equal_half_half(key_states.m_data, key_statesGT.m_data, num_heads * sqlen * head_dim);

    if (!success)
        std::cout << "-------- Test of " << __func__ << ": Fail! -------- " << std::endl;
    else
        std::cout << "-------- Test of " << __func__ << ": Passed! -------- " << std::endl;

    cudaFree(buffer_1);
    cudaFree(buffer_2);
    cudaFree(buffer_3);
    cudaFree(buffer_4);
    cudaFree(buffer_5);
    cudaFree(buffer_6);
}

void test_batch_Add_cuda() {
    const struct model_config llama7B = llama_7B;
    const int sqlen = 9, past_sqlen = 0, num_heads = llama7B.num_heads;
    const int tgz = (sqlen + past_sqlen);

    half* buffer_1;
    cudaMallocManaged(&buffer_1, sizeof(half) * num_heads * sqlen * tgz);
    Matrix3D<half> attn_weights(buffer_1, num_heads, sqlen, tgz);
    read_to_array_half("assets/llama/tests/ops/batch_Add/input_half.bin", attn_weights.m_data, num_heads * sqlen * tgz);

    half* buffer_2;
    cudaMallocManaged(&buffer_2, sizeof(half) * sqlen * tgz);
    Matrix3D<half> attention_mask(buffer_2, 1, sqlen, tgz);
    read_to_array_half("assets/llama/tests/ops/batch_Add/input2_half.bin", attention_mask.m_data, sqlen * tgz);

    half* buffer_3;
    cudaMallocManaged(&buffer_3, sizeof(half) * num_heads * sqlen * tgz);
    Matrix3D<half> output_attn_weightsGT(buffer_3, num_heads, sqlen, tgz);
    read_to_array_half("assets/llama/tests/ops/batch_Add/output_half.bin", output_attn_weightsGT.m_data, num_heads * sqlen * tgz);

    half* buffer_4;
    cudaMallocManaged(&buffer_4, sizeof(half) * num_heads * sqlen * tgz);
    Matrix3D<half> output_attn_weights(buffer_4, num_heads, sqlen, tgz);

    dim3 threadsPerBlock(8, 8, 8);
    dim3 numBlocks2((num_heads + threadsPerBlock.x - 1) / threadsPerBlock.x,
                (sqlen + threadsPerBlock.y - 1) / threadsPerBlock.y,
                (tgz + threadsPerBlock.z - 1) / threadsPerBlock.z);
    batch_Add_cuda<<<numBlocks2, threadsPerBlock>>>(attn_weights, attention_mask, output_attn_weights);
    cudaDeviceSynchronize();

    bool success = check_two_equal_half_half(output_attn_weights.m_data, output_attn_weightsGT.m_data, num_heads * sqlen * tgz);

    if (!success)
        std::cout << "-------- Test of " << __func__ << ": Fail! -------- " << std::endl;
    else
        std::cout << "-------- Test of " << __func__ << ": Passed! -------- " << std::endl;

    cudaFree(buffer_1);
    cudaFree(buffer_2);
    cudaFree(buffer_3);
    cudaFree(buffer_4);
}


int main() {
    /* CPU-version ops */
    // OPT
    test_LayerNormQ();
    test_LayerNormQ_len512();
    test_LayerNormQ_1_3B();
    test_LayerNorm();
    test_LayerNorm_1_3B_len512();
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

    /* GPU-version ops */
    test_FP16Linear_int4();
    test_LlamaRMSNorm_cuda();
    test_softmax_cuda();
    test_BMM_F16T();
    test_RotaryPosEmb_cuda();
    test_batch_Add_cuda();
    // test_Embedding_cuda();

    // Report if profiling flag is on
    Profiler::getInstance().report();
}
