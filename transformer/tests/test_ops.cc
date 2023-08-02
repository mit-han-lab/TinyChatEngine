#include <cmath>

#include "operators.h"
#include "utils.h"
#include "utils_memalloc.h"

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
    const int sqlen = 9, b = 1, embed_dim = llama7B.embed_dim;

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

void test_FPLinear_int4() {
    const int m = 1, n = 32000, k = 4096;

    MemoryAllocator mem_buf;

    Matrix3D<float> hidden_states(mem_buf.get_fpbuffer(m * k), 1, m, k);
    Matrix3D<float> weight(mem_buf.get_fpbuffer(n * k), 1, n, k);
    Matrix3D<float> outputGT(mem_buf.get_fpbuffer(m * n), 1, m, n);
    Matrix3D<float> output(mem_buf.get_fpbuffer(m * n), 1, m, n);

    hidden_states.load("assets/llama/tests/ops/Linear/input.bin");
    outputGT.load("assets/llama/tests/ops/Linear/output.bin");
    Linear_FP op(weight, "models/LLaMA_7B/lm_head.bin");

    // quantize the weight to int4
    Matrix3D<uint8_t> int4_weight((uint8_t *)mem_buf.get_int8buffer(n * k / 2), 1, n, k / 2);
    // Linear_FP_int4 int4_op;
    Linear_FP_int4 int4_op = Linear_FP_int4(int4_weight, "models/LLaMA_7B/lm_head/");

    Matrix3D<float> outputQ(mem_buf.get_fpbuffer(m * n), 1, m, n);
    Matrix3D<float> outputQ_simd(mem_buf.get_fpbuffer(m * n), 1, m, n);
    Matrix3D<float> outputQ_fast(mem_buf.get_fpbuffer(m * n), 1, m, n);

    // warm up
    for (int i = 0; i < 10; i++) {
        int4_op.forward(hidden_states, outputQ_fast);
    }

    Profiler::getInstance().reset();
    const int flops = k * m * n * 2;
    STATS_FLOPS("fp32", flops);
    op.forward(hidden_states, output);
    STATS_END("fp32");

    STATS_FLOPS("int4_ref", flops);
    int4_op.forward_ref(hidden_states, outputQ);
    STATS_END("int4_ref");

    for (int i = 0; i < 100; i++) {
        STATS_FLOPS("int4_fast", flops);
        int4_op.forward(hidden_states, outputQ_fast);
        STATS_END("int4_fast");
    }
#ifdef USE_INT8_INT4_PRODUCT
    // With a8w4, we allow larger error
    bool success = check_two_equal(outputQ.m_data, outputQ_fast.m_data, outputQ_fast.length(), 7e-4);
#else
    bool success = check_two_equal(outputQ.m_data, outputQ_fast.m_data, outputQ_fast.length(), 1e-10);
#endif

    if (!success)
        std::cout << "-------- Test of " << __func__ << ": Fail! -------- " << std::endl;
    else
        std::cout << "-------- Test of " << __func__ << ": Passed! -------- " << std::endl;
}

int main() {
    // from OPT
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
    test_FPLinear_int4();
    // Report if profiling flag is on
    Profiler::getInstance().report();
}
