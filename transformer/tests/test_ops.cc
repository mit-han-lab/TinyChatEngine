#include <cmath>

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

void test_LayerNormQ() {
    const int b = 1, m = 108, n = 768;
    MemoryAllocator mem_buf;

    float* intput_arr = mem_buf.get_fpbuffer(b * m * n);
    float* weight_arr = mem_buf.get_fpbuffer(b * n);
    float* bias_arr = mem_buf.get_fpbuffer(b * n);
    int8_t* output_arr = mem_buf.get_int8buffer(b * m * n);
    int8_t* GToutput_arr = mem_buf.get_int8buffer(b * m * n);

    Matrix3D<float> input(intput_arr, b, m, n);
    Matrix3D<float> weight(weight_arr, b, 1, n);
    Matrix3D<float> bias(bias_arr, b, 1, n);
    Matrix3D<int8_t> output(output_arr, b, m, n);
    Matrix3D<int8_t> GToutput(GToutput_arr, b, m, n);

    read_to_array((char*)"assets/LayerNormQ_bias.bin", bias_arr, b * n);
    read_to_array((char*)"assets/LayerNormQ_x.bin", intput_arr, b * m * n);
    read_to_array((char*)"assets/LayerNormQ_weight.bin", weight_arr, b * n);
    read_to_array((char*)"assets/LayerNormQ_out.bin", GToutput_arr, b * m * n);

    struct LayerNormQ_params op_params = {weight, bias};

    LayerNormQ test_op = LayerNormQ(op_params);

    test_op.forward(input, output);

    bool sucess = check_two_exact_equal(output_arr, GToutput_arr, b * m * n);
    if (!sucess)
        std::cout << "-------- Test of " << __func__ << ": Fail! -------- " << std::endl;
    else
        std::cout << "-------- Test of " << __func__ << ": Passed! -------- " << std::endl;
}

void test_LayerNormQ_len512() {
    const int b = 1, m = 512, n = 768;
    MemoryAllocator mem_buf;

    float* intput_arr = mem_buf.get_fpbuffer(b * m * n);
    float* weight_arr = mem_buf.get_fpbuffer(b * n);
    float* bias_arr = mem_buf.get_fpbuffer(b * n);
    int8_t* output_arr = mem_buf.get_int8buffer(b * m * n);
    int8_t* GToutput_arr = mem_buf.get_int8buffer(b * m * n);

    Matrix3D<float> input(intput_arr, b, m, n);
    Matrix3D<float> weight(weight_arr, b, 1, n);
    Matrix3D<float> bias(bias_arr, b, 1, n);
    Matrix3D<int8_t> output(output_arr, b, m, n);
    Matrix3D<int8_t> GToutput(GToutput_arr, b, m, n);

    // read_to_array((char*)"assets/LayerNormQ_bias.bin", bias_arr, b * n);
    read_to_array((char*)"assets/tests/OPT_125m/LayerNormQ_x_len512.bin", intput_arr, b * m * n);
    // read_to_array((char*)"assets/LayerNormQ_weight.bin", weight_arr, b * n);
    read_to_array((char*)"assets/tests/OPT_125m/LayerNormQ_y_len512.bin", GToutput_arr, b * m * n);

    struct LayerNormQ_params op_params = {weight, bias};

    LayerNormQ test_op = LayerNormQ(op_params);
    load_LayerNormQ(test_op, "models/OPT_125m/decoder/layer0/self_attn_layer_norm");

    test_op.forward(input, output);

    bool sucess = check_two_equal(output_arr, GToutput_arr, b * m * n);
    if (!sucess)
        std::cout << "-------- Test of " << __func__ << ": Fail! -------- " << std::endl;
    else
        std::cout << "-------- Test of " << __func__ << ": Passed! -------- " << std::endl;
}

void test_LayerNormQ_1_3B() {
    const int b = 1, m = 108, n = 2048;
    MemoryAllocator mem_buf;

    float* intput_arr = mem_buf.get_fpbuffer(b * m * n);
    float* weight_arr = mem_buf.get_fpbuffer(b * n);
    float* bias_arr = mem_buf.get_fpbuffer(b * n);
    int8_t* output_arr = mem_buf.get_int8buffer(b * m * n);
    int8_t* GToutput_arr = mem_buf.get_int8buffer(b * m * n);

    Matrix3D<float> input(intput_arr, b, m, n);
    Matrix3D<float> weight(weight_arr, b, 1, n);
    Matrix3D<float> bias(bias_arr, b, 1, n);
    Matrix3D<int8_t> output(output_arr, b, m, n);
    Matrix3D<int8_t> GToutput(GToutput_arr, b, m, n);

    read_to_array((char*)"assets/tests/OPT_1.3B/LayerNormQ_x.bin", intput_arr, b * m * n);
    read_to_array((char*)"assets/tests/OPT_1.3B/LayerNormQ_out.bin", GToutput_arr, b * m * n);

    struct LayerNormQ_params op_params = {weight, bias};

    LayerNormQ op = LayerNormQ(op_params);
    load_LayerNormQ(op, "models/OPT_1.3B/decoder/layer0/self_attn_layer_norm/");

    LayerNormQ test_op = LayerNormQ(op_params);

    test_op.forward(input, output);

    bool sucess = check_two_exact_equal(output_arr, GToutput_arr, b * m * n);
    if (!sucess)
        std::cout << "-------- Test of " << __func__ << ": Fail! -------- " << std::endl;
    else
        std::cout << "-------- Test of " << __func__ << ": Passed! -------- " << std::endl;
}

void test_LayerNorm() {
    const int b = 1, m = 108, n = 768;
    MemoryAllocator mem_buf;

    float* intput_arr = mem_buf.get_fpbuffer(b * m * n);
    float* weight_arr = mem_buf.get_fpbuffer(b * n);
    float* bias_arr = mem_buf.get_fpbuffer(b * n);
    float* output_arr = mem_buf.get_fpbuffer(b * m * n);
    float* GToutput_arr = mem_buf.get_fpbuffer(b * m * n);

    Matrix3D<float> input(intput_arr, b, m, n);
    Matrix3D<float> weight(weight_arr, b, 1, n);
    Matrix3D<float> bias(bias_arr, b, 1, n);
    Matrix3D<float> output(output_arr, b, m, n);
    Matrix3D<float> GToutput(GToutput_arr, b, m, n);

    read_to_array((char*)"assets/tests/decoder/final_layer_norm_hidden_states.bin", intput_arr, b * m * n);
    read_to_array((char*)"assets/tests/decoder/final_layer_norm_output.bin", GToutput_arr, b * m * n);

    struct LayerNorm_params op_params = {weight, bias};

    LayerNorm test_op = LayerNorm(op_params);
    load_LayerNorm(test_op, "models/OPT_125m/decoder/final_layer_norm/");

    test_op.forward(input, output);

    bool sucess = check_two_equal(output_arr, GToutput_arr, b * m * n);
    if (!sucess)
        std::cout << "-------- Test of " << __func__ << ": Fail! -------- " << std::endl;
    else
        std::cout << "-------- Test of " << __func__ << ": Passed! -------- " << std::endl;
}

void test_LayerNorm_1_3B_len512() {
    const int b = 1, m = 512, n = 2048;
    MemoryAllocator mem_buf;

    float* intput_arr = mem_buf.get_fpbuffer(b * m * n);
    float* weight_arr = mem_buf.get_fpbuffer(b * n);
    float* bias_arr = mem_buf.get_fpbuffer(b * n);
    float* output_arr = mem_buf.get_fpbuffer(b * m * n);
    float* GToutput_arr = mem_buf.get_fpbuffer(b * m * n);

    Matrix3D<float> input(intput_arr, b, m, n);
    Matrix3D<float> weight(weight_arr, b, 1, n);
    Matrix3D<float> bias(bias_arr, b, 1, n);
    Matrix3D<float> output(output_arr, b, m, n);
    Matrix3D<float> GToutput(GToutput_arr, b, m, n);

    read_to_array((char*)"assets/tests/OPT_1.3B/decoder/final_layer_norm_hidden_states.bin", intput_arr, b * m * n);
    read_to_array((char*)"assets/tests/OPT_1.3B/decoder/final_layer_norm_output.bin", GToutput_arr, b * m * n);

    struct LayerNorm_params op_params = {weight, bias};

    LayerNorm test_op = LayerNorm(op_params);
    load_LayerNorm(test_op, "models/OPT_1.3B/decoder/final_layer_norm/");

    test_op.forward(input, output);

    bool sucess = check_two_equal(output_arr, GToutput_arr, b * m * n, 8e-6);
    if (!sucess)
        std::cout << "-------- Test of " << __func__ << ": Fail! -------- " << std::endl;
    else
        std::cout << "-------- Test of " << __func__ << ": Passed! -------- " << std::endl;
}

void test_W8A8B8O8LinearReLU() {
    const int b = 1, m = 108, k = 768, n = 3072;
    const float alpha = 0.0005035400390625, beta = 0.02130126953125;
    MemoryAllocator mem_buf;

    int8_t* intput_arr = mem_buf.get_int8buffer(b * m * k);
    int8_t* weight_arr = mem_buf.get_int8buffer(b * k * n);
    int8_t* biasint8_arr = mem_buf.get_int8buffer(b * n);
    int8_t* output_arr = mem_buf.get_int8buffer(b * m * n);
    int8_t* GToutput_arr = mem_buf.get_int8buffer(b * m * n);

    Matrix3D<int8_t> input(intput_arr, b, m, k);
    Matrix3D<int8_t> weight(weight_arr, b, n, k);
    Matrix3D<int8_t> bias(biasint8_arr, b, 1, n);
    Matrix3D<int8_t> output(output_arr, b, m, n);
    Matrix3D<int8_t> GToutput(GToutput_arr, b, m, n);

    read_to_array((char*)"assets/tests/OPT_125m/W8A8B8O8LinearReLU_x.bin", intput_arr, b * m * k);
    read_to_array((char*)"assets/tests/OPT_125m/W8A8B8O8LinearReLU_y.bin", GToutput_arr, m * n);

    struct W8A8B8O8LinearReLU_params op_params = {weight, bias, alpha, beta};

    W8A8B8O8LinearReLU test_op = W8A8B8O8LinearReLU(op_params);
    load_W8A8B8O8LinearReLU_params(test_op, "models/OPT_125m/decoder/layer0/fc1/");

    test_op.forward(input, output);

    bool sucess = check_two_exact_equal(output_arr, GToutput_arr, m * n);
    if (!sucess)
        std::cout << "-------- Test of " << __func__ << ": Fail! -------- " << std::endl;
    else
        std::cout << "-------- Test of " << __func__ << ": Passed! -------- " << std::endl;
}

void test_W8A8B8O8LinearReLU_1_3B() {
    const int b = 1, m = 108, k = 2048, n = 8192;
    const float alpha = 0.0025501251220703125, beta = 0.0106048583984375;
    MemoryAllocator mem_buf;

    int8_t* intput_arr = mem_buf.get_int8buffer(b * m * k);
    int8_t* weight_arr = mem_buf.get_int8buffer(b * k * n);
    int8_t* biasint8_arr = mem_buf.get_int8buffer(b * n);
    int8_t* output_arr = mem_buf.get_int8buffer(b * m * n);
    int8_t* GToutput_arr = mem_buf.get_int8buffer(b * m * n);

    Matrix3D<int8_t> input(intput_arr, b, m, k);
    Matrix3D<int8_t> weight(weight_arr, b, n, k);
    Matrix3D<int8_t> bias(biasint8_arr, b, 1, n);
    Matrix3D<int8_t> output(output_arr, b, m, n);
    Matrix3D<int8_t> GToutput(GToutput_arr, b, m, n);

    read_to_array((char*)"assets/tests/OPT_1.3B/W8A8B8O8LinearReLU_x.bin", intput_arr, b * m * k);
    read_to_array((char*)"assets/tests/OPT_1.3B/W8A8B8O8LinearReLU_y.bin", GToutput_arr, m * n);

    struct W8A8B8O8LinearReLU_params op_params = {weight, bias, alpha, beta};

    W8A8B8O8LinearReLU test_op = W8A8B8O8LinearReLU(op_params);
    load_W8A8B8O8LinearReLU_params(test_op, "models/OPT_1.3B/decoder/layer0/fc1/");

    test_op.forward(input, output);

    bool sucess = check_two_exact_equal(output_arr, GToutput_arr, m * n);
    if (!sucess)
        std::cout << "-------- Test of " << __func__ << ": Fail! -------- " << std::endl;
    else
        std::cout << "-------- Test of " << __func__ << ": Passed! -------- " << std::endl;
}

void test_W8A8BFP32OFP32Linear() {
    const int b = 1, m = 512, k = 768, n = 768;
    const float alpha = 0.00004565715789794922;
    MemoryAllocator mem_buf;

    int8_t* intput_arr = mem_buf.get_int8buffer(b * m * k);
    int8_t* weight_arr = mem_buf.get_int8buffer(b * k * n);
    float* bias_arr = mem_buf.get_fpbuffer(b * n);
    float* output_arr = mem_buf.get_fpbuffer(b * m * n);
    float* GToutput_arr = mem_buf.get_fpbuffer(b * m * n);

    Matrix3D<int8_t> input(intput_arr, b, m, k);
    Matrix3D<int8_t> weight(weight_arr, b, n, k);
    Matrix3D<float> bias(bias_arr, b, 1, n);
    Matrix3D<float> output(output_arr, b, m, n);
    Matrix3D<float> GToutput(GToutput_arr, b, m, n);

    read_to_array((char*)"assets/tests/OPT_125m/W8A8BFP32OFP32Linear_x.bin", intput_arr, b * m * k);
    read_to_array((char*)"assets/tests/OPT_125m/W8A8BFP32OFP32Linear_y.bin", GToutput_arr, b * m * n);

    struct W8A8BFP32OFP32Linear_params op_params = {weight, bias, alpha};

    auto test_op = W8A8BFP32OFP32Linear(op_params);
    load_W8A8BFP32OFP32Linear_params(test_op, "models/OPT_125m/decoder/layer0/self_attn/out_proj/");
    test_op.forward(input, output);

    bool sucess = check_two_equal(output_arr, GToutput_arr, b * m * n);
    if (!sucess)
        std::cout << "-------- Test of " << __func__ << ": Fail! -------- " << std::endl;
    else
        std::cout << "-------- Test of " << __func__ << ": Passed! -------- " << std::endl;
}

void test_W8A8BFP32OFP32Linear_1_3B() {
    const int b = 1, m = 108, k = 2048, n = 2048;
    const float alpha = 0.00012445449829101562;
    MemoryAllocator mem_buf;

    int8_t* intput_arr = mem_buf.get_int8buffer(b * m * k);
    int8_t* weight_arr = mem_buf.get_int8buffer(b * k * n);
    float* bias_arr = mem_buf.get_fpbuffer(b * n);
    float* output_arr = mem_buf.get_fpbuffer(b * m * n);
    float* GToutput_arr = mem_buf.get_fpbuffer(b * m * n);

    Matrix3D<int8_t> input(intput_arr, b, m, k);
    Matrix3D<int8_t> weight(weight_arr, b, n, k);
    Matrix3D<float> bias(bias_arr, b, 1, n);
    Matrix3D<float> output(output_arr, b, m, n);
    Matrix3D<float> GToutput(GToutput_arr, b, m, n);

    read_to_array((char*)"assets/tests/OPT_1.3B/W8A8BFP32OFP32Linear_x.bin", intput_arr, b * m * k);
    read_to_array((char*)"assets/tests/OPT_1.3B/W8A8BFP32OFP32Linear_y.bin", GToutput_arr, m * n);

    struct W8A8BFP32OFP32Linear_params op_params = {weight, bias, alpha};

    auto test_op = W8A8BFP32OFP32Linear(op_params);
    load_W8A8BFP32OFP32Linear_params(test_op, "models/OPT_1.3B/decoder/layer0/self_attn/out_proj/");
    test_op.forward(input, output);

    bool sucess = check_two_equal(output_arr, GToutput_arr, b * m * n);
    if (!sucess)
        std::cout << "-------- Test of " << __func__ << ": Fail! -------- " << std::endl;
    else
        std::cout << "-------- Test of " << __func__ << ": Passed! -------- " << std::endl;
}

void test_W8A8B8O8Linear() {
    const int b = 1, m = 108, k = 768, n = 768;
    MemoryAllocator mem_buf;

    int8_t* intput_arr = mem_buf.get_int8buffer(b * m * k);
    int8_t* weight_arr = mem_buf.get_int8buffer(b * k * n);
    int8_t* weightGT_arr = mem_buf.get_int8buffer(b * k * n);
    int8_t* biasint8_arr = mem_buf.get_int8buffer(b * n);
    int8_t* biasint8GT_arr = mem_buf.get_int8buffer(b * n);
    int32_t* biasint32_arr = mem_buf.get_intbuffer(b * n);
    int8_t* output_arr = mem_buf.get_int8buffer(b * m * n);
    int8_t* GToutput_arr = mem_buf.get_int8buffer(b * m * n);

    Matrix3D<int8_t> input(intput_arr, b, m, k);
    Matrix3D<int8_t> weight(weight_arr, b, n, k);
    Matrix3D<int8_t> bias(biasint8_arr, b, 1, n);
    Matrix3D<int8_t> output(output_arr, b, m, n);
    Matrix3D<int8_t> GToutput(GToutput_arr, b, m, n);

    read_to_array((char*)"assets/tests/OPT_125m/W8A8B8O8Linear_x.bin", intput_arr, b * m * k);
    read_to_array((char*)"assets/tests/OPT_125m/W8A8B8O8Linear_y.bin", GToutput_arr, b * m * n);

    struct W8A8B8O8Linear_params op_params = {weight, bias};

    W8A8B8O8Linear test_op = W8A8B8O8Linear(op_params);
    load_W8A8B8O8Linear_params(test_op, "models/OPT_125m/decoder/layer0/self_attn/q_proj/");

    test_op.forward(input, output);

    bool sucess = check_two_exact_equal(output_arr, GToutput_arr, b * m * n);
    if (!sucess)
        std::cout << "-------- Test of " << __func__ << ": Fail! -------- " << std::endl;
    else
        std::cout << "-------- Test of " << __func__ << ": Passed! -------- " << std::endl;
}

void test_W8A8B8O8Linear_1_3B() {
    const int b = 1, m = 108, k = 2048, n = 2048;
    MemoryAllocator mem_buf;

    int8_t* intput_arr = mem_buf.get_int8buffer(b * m * k);
    int8_t* weight_arr = mem_buf.get_int8buffer(b * k * n);
    int8_t* biasint8_arr = mem_buf.get_int8buffer(b * n);
    int8_t* output_arr = mem_buf.get_int8buffer(b * m * n);
    int8_t* GToutput_arr = mem_buf.get_int8buffer(b * m * n);

    Matrix3D<int8_t> input(intput_arr, b, m, k);
    Matrix3D<int8_t> weight(weight_arr, b, n, k);
    Matrix3D<int8_t> bias(biasint8_arr, b, 1, n);
    Matrix3D<int8_t> output(output_arr, b, m, n);
    Matrix3D<int8_t> GToutput(GToutput_arr, b, m, n);

    read_to_array((char*)"assets/tests/OPT_1.3B/W8A8B8O8Linear_x.bin", intput_arr, b * m * k);
    read_to_array((char*)"assets/tests/OPT_1.3B/W8A8B8O8Linear_y.bin", GToutput_arr, m * n);

    struct W8A8B8O8Linear_params op_params = {weight, bias};

    W8A8B8O8Linear test_op = W8A8B8O8Linear(op_params);
    load_W8A8B8O8Linear_params(test_op, "models/OPT_1.3B/decoder/layer0/self_attn/q_proj/");

    test_op.forward(input, output);

    bool sucess = check_two_exact_equal(output_arr, GToutput_arr, b * m * n);
    if (!sucess)
        std::cout << "-------- Test of " << __func__ << ": Fail! -------- " << std::endl;
    else
        std::cout << "-------- Test of " << __func__ << ": Passed! -------- " << std::endl;
}

void test_BMM_S8T_S8N_F32T() {
    const int b = 12, m = 512, k = 64, n = 512;
    const float alpha = 0.0006456375122070312;
    MemoryAllocator mem_buf;

    int8_t* intput_arr = mem_buf.get_int8buffer(b * m * k);
    int8_t* weight_arr = mem_buf.get_int8buffer(b * k * n);
    float* output_arr = mem_buf.get_fpbuffer(b * m * n);
    float* GToutput_arr = mem_buf.get_fpbuffer(b * m * n);

    Matrix3D<int8_t> input(intput_arr, b, m, k);
    Matrix3D<int8_t> weight(weight_arr, b, n, k);
    Matrix3D<float> output(output_arr, b, m, n);
    Matrix3D<float> GToutput(GToutput_arr, b, m, n);

    read_to_array((char*)"assets/BMM_S8T_S8N_F32T_x.bin", intput_arr, b * m * k);
    read_to_array((char*)"assets/BMM_S8T_S8N_F32T_weight.bin", weight_arr, b * n * k);
    read_to_array((char*)"assets/BMM_S8T_S8N_F32T_y.bin", GToutput_arr, b * m * n);

    struct BMM_S8T_S8N_F32T_params op_params = {alpha};

    auto test_op = BMM_S8T_S8N_F32T(op_params);
    test_op.forward(input, weight, output);

    bool sucess = check_two_equal(output_arr, GToutput_arr, b * m * n);

    if (!sucess)
        std::cout << "-------- Test of " << __func__ << ": Fail! -------- " << std::endl;
    else
        std::cout << "-------- Test of " << __func__ << ": Passed! -------- " << std::endl;
}

void test_BMM_S8T_S8N_F32T_1_3B() {
    const int heads = 32, sqlen = 108, c = 64;
    MemoryAllocator mem_buf;

    int8_t* intput_arr = mem_buf.get_int8buffer(heads * sqlen * c);
    int8_t* weight_arr = mem_buf.get_int8buffer(heads * sqlen * c);
    float* output_arr = mem_buf.get_fpbuffer(heads * sqlen * sqlen);
    float* GToutput_arr = mem_buf.get_fpbuffer(heads * sqlen * sqlen);

    Matrix3D<int8_t> input(intput_arr, heads, sqlen, c);
    Matrix3D<int8_t> weight(weight_arr, heads, sqlen, c);
    Matrix3D<float> output(output_arr, heads, sqlen, sqlen);
    Matrix3D<float> GToutput(GToutput_arr, heads, sqlen, sqlen);

    read_to_array((char*)"assets/tests/OPT_1.3B/BMM_S8T_S8N_F32T_x1.bin", intput_arr, input.length());
    read_to_array((char*)"assets/tests/OPT_1.3B/BMM_S8T_S8N_F32T_x2.bin", weight_arr, weight.length());
    read_to_array((char*)"assets/tests/OPT_1.3B/BMM_S8T_S8N_F32T_y.bin", GToutput_arr, GToutput.length());

    struct BMM_S8T_S8N_F32T_params op_params = {};

    auto test_op = BMM_S8T_S8N_F32T(op_params);
    load_BMM_S8T_S8N_F32T(test_op, "models/OPT_1.3B/decoder/layer0/self_attn/qk_bmm/");
    test_op.forward(input, weight, output);

    bool sucess = check_two_equal(output_arr, GToutput_arr, GToutput.length());

    if (!sucess)
        std::cout << "-------- Test of " << __func__ << ": Fail! -------- " << std::endl;
    else
        std::cout << "-------- Test of " << __func__ << ": Passed! -------- " << std::endl;
}

void test_BMM_S8T_S8N_S8T() {
    const int b = 12, m = 512, k = 512, n = 64;
    const float alpha = 0.013031005859375;
    MemoryAllocator mem_buf;

    int8_t* intput_arr = mem_buf.get_int8buffer(b * m * k);
    int8_t* weight_arr = mem_buf.get_int8buffer(b * k * n);
    int8_t* output_arr = mem_buf.get_int8buffer(b * m * n);
    int8_t* GToutput_arr = mem_buf.get_int8buffer(b * m * n);

    Matrix3D<int8_t> input(intput_arr, b, m, k);
    Matrix3D<int8_t> weight(weight_arr, b, n, k);
    Matrix3D<int8_t> output(output_arr, b, m, n);
    Matrix3D<int8_t> GToutput(GToutput_arr, b, m, n);

    read_to_array((char*)"assets/BMM_S8T_S8N_S8T_x.bin", intput_arr, b * m * k);
    read_to_array((char*)"assets/BMM_S8T_S8N_S8T_weight.bin", weight_arr, b * n * k);
    read_to_array((char*)"assets/BMM_S8T_S8N_S8T_y.bin", GToutput_arr, b * m * n);

    struct BMM_S8T_S8N_S8T_params op_params = {alpha};

    auto test_op = BMM_S8T_S8N_S8T(op_params);
    test_op.forward(input, weight, output);

    bool sucess = check_two_exact_equal(output_arr, GToutput_arr, GToutput.length());
    if (!sucess)
        std::cout << "-------- Test of " << __func__ << ": Fail! -------- " << std::endl;
    else
        std::cout << "-------- Test of " << __func__ << ": Passed! -------- " << std::endl;
}

void test_BMM_S8T_S8N_S8T_1_3B() {
    const int heads = 32, sqlen = 108, c = 64;
    const float alpha = 0.00787353515625;
    MemoryAllocator mem_buf;

    int8_t* intput_arr = mem_buf.get_int8buffer(heads * sqlen * sqlen);
    int8_t* weight_arr = mem_buf.get_int8buffer(heads * c * sqlen);
    int8_t* output_arr = mem_buf.get_int8buffer(heads * sqlen * c);
    int8_t* GToutput_arr = mem_buf.get_int8buffer(heads * sqlen * c);

    Matrix3D<int8_t> input(intput_arr, heads, sqlen, sqlen);
    Matrix3D<int8_t> weight(weight_arr, heads, c, sqlen);
    Matrix3D<int8_t> output(output_arr, heads, sqlen, c);
    Matrix3D<int8_t> GToutput(GToutput_arr, heads, sqlen, c);

    read_to_array((char*)"assets/tests/OPT_1.3B/BMM_S8T_S8N_S8T_x1.bin", intput_arr, input.length());
    read_to_array((char*)"assets/tests/OPT_1.3B/BMM_S8T_S8N_S8T_x2.bin", weight_arr, weight.length());
    read_to_array((char*)"assets/tests/OPT_1.3B/BMM_S8T_S8N_S8T_y.bin", GToutput_arr, GToutput.length());

    struct BMM_S8T_S8N_S8T_params op_params = {alpha};

    auto test_op = BMM_S8T_S8N_S8T(op_params);
    test_op.forward(input, weight, output);

    bool sucess = check_two_exact_equal(output_arr, GToutput_arr, GToutput.length());
    if (!sucess)
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

    read_to_array((char*)"assets/input_ids.bin", input.m_data, sqlen);
    read_to_array((char*)"assets/inputs_embeds.bin", outputGT.m_data, sqlen * embed_dim);

    auto embed_tokens = Embedding(embed_dim, voc_size, padding_idx, weight);
    load_Embedding_params(embed_tokens, "assets/decoder/embed_tokens");

    embed_tokens.forward(input, output);
    assert(check_two_equal(output.m_data, outputGT.m_data, sqlen * embed_dim));

    std::cout << "-------- Test of " << __func__ << ": Passed! -------- " << std::endl;
}

void test_Embedding_1_3B() {
    const int voc_size = 50272, embed_dim = 2048, sqlen = 108, padding_idx = 1;
    MemoryAllocator mem_buf;

    Matrix3D<int> input(mem_buf.get_intbuffer(sqlen), 1, 1, sqlen);
    Matrix3D<float> weight(mem_buf.get_fpbuffer(voc_size * embed_dim), 1, voc_size, embed_dim);
    Matrix3D<float> output(mem_buf.get_fpbuffer(sqlen * embed_dim), 1, sqlen, embed_dim);
    Matrix3D<float> outputGT(mem_buf.get_fpbuffer(sqlen * embed_dim), 1, sqlen, embed_dim);

    read_to_array((char*)"assets/input_ids.bin", input.m_data, sqlen);
    read_to_array((char*)"assets/tests/OPT_1.3B/inputs_embeds.bin", outputGT.m_data, sqlen * embed_dim);

    auto embed_tokens = Embedding(embed_dim, voc_size, padding_idx, weight);
    load_Embedding_params(embed_tokens, "models/OPT_1.3B/decoder/embed_tokens");

    embed_tokens.forward(input, output);
    assert(check_two_equal(output.m_data, outputGT.m_data, sqlen * embed_dim));

    std::cout << "-------- Test of " << __func__ << ": Passed! -------- " << std::endl;
}

int main() {
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
}
