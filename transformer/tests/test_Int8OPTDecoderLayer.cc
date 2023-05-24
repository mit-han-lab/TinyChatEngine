#include "Int8OPTDecoder.h"
#include "operators.h"
#include "utils.h"

#define MAX_TEST_MEMORY_BUF 1 * 1024 * 1024 * 1576  // 1.5 GB
static char buffer[MAX_TEST_MEMORY_BUF];

class MemoryAllocator {
   public:
    MemoryAllocator() { this->counter = 0; }
    float *get_fpbuffer(int size) {
        int byte_size = size * sizeof(float);
        if (this->counter + byte_size > MAX_TEST_MEMORY_BUF) {
            throw("Memory allocation fails! Test case uses too much memory.");
        }
        int cur_counter = counter;
        this->counter += ((byte_size + 3) / 4) * 4;
        return (float *)&buffer[cur_counter];
    }
    int8_t *get_int8buffer(int size) {
        int byte_size = size * sizeof(int8_t);
        if (this->counter + byte_size > MAX_TEST_MEMORY_BUF) {
            throw("Memory allocation fails! Test case uses too much memory.");
        }
        int cur_counter = counter;
        this->counter += ((byte_size + 3) / 4) * 4;
        return (int8_t *)&buffer[cur_counter];
    }
    int *get_intbuffer(int size) {
        int byte_size = size * sizeof(int);
        if (this->counter + byte_size > MAX_TEST_MEMORY_BUF) {
            throw("Memory allocation fails! Test case uses too much memory.");
        }
        int cur_counter = counter;
        this->counter += ((byte_size + 3) / 4) * 4;
        return (int *)&buffer[cur_counter];
    }

   private:
    int counter;
};

void test_DecoderLayer_generate() {
    const int num_heads = 12, embed_dim = 768, sqlen = 108, b = 1, hidden_dim = 3072;
    MemoryAllocator mem_buf;

    struct BMM_S8T_S8N_F32T_params qk_bmm;
    struct BMM_S8T_S8N_S8T_params pv_bmm;
    struct W8A8B8O8Linear_params k_proj, v_proj, q_proj;
    Matrix3D<int8_t> k_proj_weight(mem_buf.get_int8buffer(embed_dim * embed_dim), 1, embed_dim, embed_dim);
    Matrix3D<int8_t> k_proj_bias(mem_buf.get_int8buffer(embed_dim), 1, 1, embed_dim);
    k_proj.weight = k_proj_weight;
    k_proj.bias = k_proj_bias;
    auto k_proj_op = W8A8B8O8Linear(k_proj);
    Matrix3D<int8_t> v_proj_weight(mem_buf.get_int8buffer(embed_dim * embed_dim), 1, embed_dim, embed_dim);
    Matrix3D<int8_t> v_proj_bias(mem_buf.get_int8buffer(embed_dim), 1, 1, embed_dim);
    v_proj.weight = v_proj_weight;
    v_proj.bias = v_proj_bias;
    auto v_proj_op = W8A8B8O8Linear(v_proj);
    Matrix3D<int8_t> q_proj_weight(mem_buf.get_int8buffer(embed_dim * embed_dim), 1, embed_dim, embed_dim);
    Matrix3D<int8_t> q_proj_bias(mem_buf.get_int8buffer(embed_dim), 1, 1, embed_dim);
    q_proj.weight = q_proj_weight;
    q_proj.bias = q_proj_bias;
    auto q_proj_op = W8A8B8O8Linear(q_proj);

    struct W8A8BFP32OFP32Linear_params out_proj;
    Matrix3D<int8_t> out_proj_weight(mem_buf.get_int8buffer(embed_dim * embed_dim), 1, embed_dim, embed_dim);
    Matrix3D<float> out_proj_bias(mem_buf.get_fpbuffer(embed_dim), 1, 1, embed_dim);
    out_proj.weight = out_proj_weight;
    out_proj.bias = out_proj_bias;
    auto out_proj_op = W8A8BFP32OFP32Linear(out_proj);

    struct LayerNormQ_params self_attn_layer_norm, final_layer_norm;
    Matrix3D<float> self_attn_layer_norm_weight(mem_buf.get_fpbuffer(embed_dim), 1, 1, embed_dim);
    Matrix3D<float> self_attn_layer_norm_bias(mem_buf.get_fpbuffer(embed_dim), 1, 1, embed_dim);
    self_attn_layer_norm.weight = self_attn_layer_norm_weight;
    self_attn_layer_norm.bias = self_attn_layer_norm_bias;

    Matrix3D<float> final_layer_norm_weight(mem_buf.get_fpbuffer(embed_dim), 1, 1, embed_dim);
    Matrix3D<float> final_layer_norm_bias(mem_buf.get_fpbuffer(embed_dim), 1, 1, embed_dim);
    final_layer_norm.weight = final_layer_norm_weight;
    final_layer_norm.bias = final_layer_norm_bias;
    LayerNormQ self_attn_layer_norm_op = LayerNormQ(self_attn_layer_norm);
    LayerNormQ final_layer_norm_op = LayerNormQ(final_layer_norm);

    struct W8A8B8O8LinearReLU_params fc1;
    Matrix3D<int8_t> fc1_weight(mem_buf.get_int8buffer(embed_dim * hidden_dim), 1, hidden_dim, embed_dim);
    Matrix3D<int8_t> fc1_bias(mem_buf.get_int8buffer(hidden_dim), 1, 1, hidden_dim);
    fc1.weight = fc1_weight;
    fc1.bias_int8 = fc1_bias;
    auto fc1_op = W8A8B8O8LinearReLU(fc1);

    struct W8A8BFP32OFP32Linear_params fc2;
    Matrix3D<int8_t> fc2_weight(mem_buf.get_int8buffer(embed_dim * hidden_dim), 1, embed_dim, hidden_dim);
    Matrix3D<float> fc2_bias(mem_buf.get_fpbuffer(embed_dim), 1, 1, embed_dim);
    fc2.weight = fc2_weight;
    fc2.bias = fc2_bias;
    auto fc2_op = W8A8BFP32OFP32Linear(fc2);

    auto qk_bmm_op = BMM_S8T_S8N_F32T(qk_bmm);
    auto pv_bmm_op = BMM_S8T_S8N_S8T(pv_bmm);

    int layer_idx = 0;
    Int8OPTDecoderLayer layer = Int8OPTDecoderLayer(
        "models/OPT_125m/decoder/layer0", get_opt_model_config(OPT_125M), layer_idx, self_attn_layer_norm_op,
        final_layer_norm_op, fc1_op, fc2_op, qk_bmm_op, pv_bmm_op, k_proj_op, v_proj_op, q_proj_op, out_proj_op);

    Matrix3D<float> hidden_states(mem_buf.get_fpbuffer(b * sqlen * embed_dim), b, sqlen, embed_dim);
    read_to_array("assets/tests/OPT_125m/generate_layer0_hidden_states.bin", hidden_states.m_data,
                  b * sqlen * embed_dim);
    Matrix3D<float> attention_mask(mem_buf.get_fpbuffer(b * sqlen * sqlen), b, sqlen, sqlen);
    read_to_array("assets/tests/OPT_125m/generate_layer0_attention_mask.bin", attention_mask.m_data, b * sqlen * sqlen);
    struct Int8OPTDecoderLayer_input input = {hidden_states, attention_mask};

    struct Int8OPTDecoderLayer_output output = layer.forward(input);

    Matrix3D<float> residualGT(mem_buf.get_fpbuffer(b * sqlen * embed_dim), b, sqlen, embed_dim);
    read_to_array("assets/tests/OPT_125m/generate_layer0_residual.bin", residualGT.m_data, b * sqlen * embed_dim);
    int8_t *key_statesGT = mem_buf.get_int8buffer(output.past_key_value.first.length());
    read_to_array("assets/tests/OPT_125m/present_key.bin", key_statesGT, output.past_key_value.first.length());
    int8_t *value_statesGT = mem_buf.get_int8buffer(output.past_key_value.second.length());
    read_to_array("assets/tests/OPT_125m/present_value.bin", value_statesGT, output.past_key_value.second.length());

    bool sucess = check_two_equal(residualGT.m_data, output.hidden_states.m_data, b * sqlen * embed_dim);
    sucess &=
        check_two_exact_equal(output.past_key_value.first.m_data, key_statesGT, output.past_key_value.first.length());
    sucess &= check_two_exact_equal(output.past_key_value.second.m_data, value_statesGT,
                                    output.past_key_value.second.length());
    if (!sucess)
        std::cout << "-------- Test of " << __func__ << ": Fail! -------- " << std::endl;
    else
        std::cout << "-------- Test of " << __func__ << ": Passed! -------- " << std::endl;
}

void test_DecoderLayer_generate_1_3B() {
    auto config = get_opt_model_config(OPT_1_3B);
    const int num_heads = config.num_heads, embed_dim = config.embed_dim, hidden_dim = config.hidden_dim;
    const int sqlen = 108, b = 1;
    MemoryAllocator mem_buf;

    struct BMM_S8T_S8N_F32T_params qk_bmm;
    struct BMM_S8T_S8N_S8T_params pv_bmm;
    struct W8A8B8O8Linear_params k_proj, v_proj, q_proj;
    Matrix3D<int8_t> k_proj_weight(mem_buf.get_int8buffer(embed_dim * embed_dim), 1, embed_dim, embed_dim);
    Matrix3D<int8_t> k_proj_bias(mem_buf.get_int8buffer(embed_dim), 1, 1, embed_dim);
    k_proj.weight = k_proj_weight;
    k_proj.bias = k_proj_bias;
    auto k_proj_op = W8A8B8O8Linear(k_proj);
    Matrix3D<int8_t> v_proj_weight(mem_buf.get_int8buffer(embed_dim * embed_dim), 1, embed_dim, embed_dim);
    Matrix3D<int8_t> v_proj_bias(mem_buf.get_int8buffer(embed_dim), 1, 1, embed_dim);
    v_proj.weight = v_proj_weight;
    v_proj.bias = v_proj_bias;
    auto v_proj_op = W8A8B8O8Linear(v_proj);
    Matrix3D<int8_t> q_proj_weight(mem_buf.get_int8buffer(embed_dim * embed_dim), 1, embed_dim, embed_dim);
    Matrix3D<int8_t> q_proj_bias(mem_buf.get_int8buffer(embed_dim), 1, 1, embed_dim);
    q_proj.weight = q_proj_weight;
    q_proj.bias = q_proj_bias;
    auto q_proj_op = W8A8B8O8Linear(q_proj);

    struct W8A8BFP32OFP32Linear_params out_proj;
    Matrix3D<int8_t> out_proj_weight(mem_buf.get_int8buffer(embed_dim * embed_dim), 1, embed_dim, embed_dim);
    Matrix3D<float> out_proj_bias(mem_buf.get_fpbuffer(embed_dim), 1, 1, embed_dim);
    out_proj.weight = out_proj_weight;
    out_proj.bias = out_proj_bias;
    auto out_proj_op = W8A8BFP32OFP32Linear(out_proj);

    struct LayerNormQ_params self_attn_layer_norm, final_layer_norm;
    Matrix3D<float> self_attn_layer_norm_weight(mem_buf.get_fpbuffer(embed_dim), 1, 1, embed_dim);
    Matrix3D<float> self_attn_layer_norm_bias(mem_buf.get_fpbuffer(embed_dim), 1, 1, embed_dim);
    self_attn_layer_norm.weight = self_attn_layer_norm_weight;
    self_attn_layer_norm.bias = self_attn_layer_norm_bias;

    Matrix3D<float> final_layer_norm_weight(mem_buf.get_fpbuffer(embed_dim), 1, 1, embed_dim);
    Matrix3D<float> final_layer_norm_bias(mem_buf.get_fpbuffer(embed_dim), 1, 1, embed_dim);
    final_layer_norm.weight = final_layer_norm_weight;
    final_layer_norm.bias = final_layer_norm_bias;
    LayerNormQ self_attn_layer_norm_op = LayerNormQ(self_attn_layer_norm);
    LayerNormQ final_layer_norm_op = LayerNormQ(final_layer_norm);

    struct W8A8B8O8LinearReLU_params fc1;
    Matrix3D<int8_t> fc1_weight(mem_buf.get_int8buffer(embed_dim * hidden_dim), 1, hidden_dim, embed_dim);
    Matrix3D<int8_t> fc1_bias(mem_buf.get_int8buffer(hidden_dim), 1, 1, hidden_dim);
    fc1.weight = fc1_weight;
    fc1.bias_int8 = fc1_bias;
    auto fc1_op = W8A8B8O8LinearReLU(fc1);

    struct W8A8BFP32OFP32Linear_params fc2;
    Matrix3D<int8_t> fc2_weight(mem_buf.get_int8buffer(embed_dim * hidden_dim), 1, embed_dim, hidden_dim);
    Matrix3D<float> fc2_bias(mem_buf.get_fpbuffer(embed_dim), 1, 1, embed_dim);
    fc2.weight = fc2_weight;
    fc2.bias = fc2_bias;
    auto fc2_op = W8A8BFP32OFP32Linear(fc2);

    auto qk_bmm_op = BMM_S8T_S8N_F32T(qk_bmm);
    auto pv_bmm_op = BMM_S8T_S8N_S8T(pv_bmm);

    int layer_idx = 0;
    Int8OPTDecoderLayer layer = Int8OPTDecoderLayer("models/OPT_1.3B/decoder/layer0", config, layer_idx,
                                                    self_attn_layer_norm_op, final_layer_norm_op, fc1_op, fc2_op,
                                                    qk_bmm_op, pv_bmm_op, k_proj_op, v_proj_op, q_proj_op, out_proj_op);

    Matrix3D<float> hidden_states(mem_buf.get_fpbuffer(b * sqlen * embed_dim), b, sqlen, embed_dim);
    read_to_array("assets/tests/OPT_1.3B/generate_layer0_hidden_states.bin", hidden_states.m_data,
                  b * sqlen * embed_dim);
    Matrix3D<float> attention_mask(mem_buf.get_fpbuffer(b * sqlen * sqlen), b, sqlen, sqlen);
    read_to_array("assets/tests/OPT_1.3B/generate_layer0_attention_mask.bin", attention_mask.m_data, b * sqlen * sqlen);
    struct Int8OPTDecoderLayer_input input = {hidden_states, attention_mask};

    struct Int8OPTDecoderLayer_output output = layer.forward(input);

    Matrix3D<float> residualGT(mem_buf.get_fpbuffer(b * sqlen * embed_dim), b, sqlen, embed_dim);
    read_to_array("assets/tests/OPT_1.3B/generate_layer0_residual.bin", residualGT.m_data, b * sqlen * embed_dim);
    int8_t *key_statesGT = mem_buf.get_int8buffer(output.past_key_value.first.length());
    read_to_array("assets/tests/OPT_1.3B/present_key.bin", key_statesGT, output.past_key_value.first.length());
    int8_t *value_statesGT = mem_buf.get_int8buffer(output.past_key_value.second.length());
    read_to_array("assets/tests/OPT_1.3B/present_value.bin", value_statesGT, output.past_key_value.second.length());

    bool sucess = check_two_equal(residualGT.m_data, output.hidden_states.m_data, b * sqlen * embed_dim);
    sucess &=
        check_two_exact_equal(output.past_key_value.first.m_data, key_statesGT, output.past_key_value.first.length());
    sucess &= check_two_exact_equal(output.past_key_value.second.m_data, value_statesGT,
                                    output.past_key_value.second.length());
    if (!sucess)
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
    read_to_array("assets/tests/OPT_125m/test_cache_hidden_states.bin", hidden_states.m_data, b * sqlen * embed_dim);
    Matrix3D<float> attention_mask(mem_buf.get_fpbuffer(b * sqlen * tgz), b, sqlen, tgz);
    read_to_array("assets/tests/OPT_125m/test_cache_causal_attention_mask.bin", attention_mask.m_data, b * sqlen * tgz);
    Matrix3D<int8_t> past_keys(mem_buf.get_int8buffer(b * past_len * embed_dim), num_heads, past_len, head_dim);
    read_to_array("assets/tests/OPT_125m/test_cache_past_key.bin", past_keys.m_data, past_keys.length());
    Matrix3D<int8_t> past_value(mem_buf.get_int8buffer(b * past_len * embed_dim), num_heads, past_len, head_dim);
    read_to_array("assets/tests/OPT_125m/test_cache_past_value.bin", past_value.m_data, past_keys.length());

    struct Int8OPTDecoderLayer_input input = {hidden_states, attention_mask, past_keys, past_value};

    struct Int8OPTDecoderLayer_output output = decoder.layers[0].forward(input);

    Matrix3D<float> residualGT(mem_buf.get_fpbuffer(b * sqlen * embed_dim), b, sqlen, embed_dim);
    read_to_array("assets/tests/OPT_125m/test_cache_residual.bin", residualGT.m_data, b * sqlen * embed_dim);
    // // print_first_k_elelment("output.hidden_states.m_data", output.hidden_states.m_data, 64);
    // // print_first_k_elelment("residualGT.m_data", residualGT.m_data, 64);
    int8_t *key_statesGT = mem_buf.get_int8buffer(output.past_key_value.first.length());
    read_to_array("assets/tests/OPT_125m/test_present_key.bin", key_statesGT, output.past_key_value.first.length());
    int8_t *value_statesGT = mem_buf.get_int8buffer(output.past_key_value.second.length());
    read_to_array("assets/tests/OPT_125m/test_present_value.bin", value_statesGT,
                  output.past_key_value.second.length());

    bool sucess = check_two_equal(residualGT.m_data, output.hidden_states.m_data, b * sqlen * embed_dim);
    sucess &=
        check_two_exact_equal(output.past_key_value.first.m_data, key_statesGT, output.past_key_value.first.length());
    sucess &= check_two_exact_equal(output.past_key_value.second.m_data, value_statesGT,
                                    output.past_key_value.second.length());
    if (!sucess)
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
    read_to_array("assets/tests/OPT_1.3B/test_cache_hidden_states.bin", hidden_states.m_data, b * sqlen * embed_dim);
    Matrix3D<float> attention_mask(mem_buf.get_fpbuffer(b * sqlen * tgz), b, sqlen, tgz);
    read_to_array("assets/tests/OPT_1.3B/test_cache_causal_attention_mask.bin", attention_mask.m_data, b * sqlen * tgz);
    Matrix3D<int8_t> past_keys(mem_buf.get_int8buffer(b * past_len * embed_dim), num_heads, past_len, head_dim);
    read_to_array("assets/tests/OPT_1.3B/test_cache_past_key.bin", past_keys.m_data, past_keys.length());
    Matrix3D<int8_t> past_value(mem_buf.get_int8buffer(b * past_len * embed_dim), num_heads, past_len, head_dim);
    read_to_array("assets/tests/OPT_1.3B/test_cache_past_value.bin", past_value.m_data, past_keys.length());

    struct Int8OPTDecoderLayer_input input = {hidden_states, attention_mask, past_keys, past_value};

    struct Int8OPTDecoderLayer_output output = decoder.layers[0].forward(input);

    Matrix3D<float> residualGT(mem_buf.get_fpbuffer(b * sqlen * embed_dim), b, sqlen, embed_dim);
    read_to_array("assets/tests/OPT_1.3B/test_cache_residual.bin", residualGT.m_data, b * sqlen * embed_dim);
    int8_t *key_statesGT = mem_buf.get_int8buffer(output.past_key_value.first.length());
    read_to_array("assets/tests/OPT_1.3B/test_present_key.bin", key_statesGT, output.past_key_value.first.length());
    int8_t *value_statesGT = mem_buf.get_int8buffer(output.past_key_value.second.length());
    read_to_array("assets/tests/OPT_1.3B/test_present_value.bin", value_statesGT,
                  output.past_key_value.second.length());

    bool sucess = check_two_equal(residualGT.m_data, output.hidden_states.m_data, b * sqlen * embed_dim);
    sucess &=
        check_two_exact_equal(output.past_key_value.first.m_data, key_statesGT, output.past_key_value.first.length());
    sucess &= check_two_exact_equal(output.past_key_value.second.m_data, value_statesGT,
                                    output.past_key_value.second.length());
    if (!sucess)
        std::cout << "-------- Test of " << __func__ << ": Fail! -------- " << std::endl;
    else
        std::cout << "-------- Test of " << __func__ << ": Passed! -------- " << std::endl;
}

void test_DecoderLayer() {
    auto config = get_opt_model_config(OPT_125M);
    const int num_heads = config.num_heads, embed_dim = config.embed_dim, hidden_dim = config.hidden_dim;
    const int sqlen = 512, b = 1, head_dim = embed_dim / num_heads;
    MemoryAllocator mem_buf;

    struct BMM_S8T_S8N_F32T_params qk_bmm;
    struct BMM_S8T_S8N_S8T_params pv_bmm;
    struct W8A8B8O8Linear_params k_proj, v_proj, q_proj;
    Matrix3D<int8_t> k_proj_weight(mem_buf.get_int8buffer(embed_dim * embed_dim), 1, embed_dim, embed_dim);
    Matrix3D<int8_t> k_proj_bias(mem_buf.get_int8buffer(embed_dim), 1, 1, embed_dim);
    k_proj.weight = k_proj_weight;
    k_proj.bias = k_proj_bias;
    auto k_proj_op = W8A8B8O8Linear(k_proj);
    Matrix3D<int8_t> v_proj_weight(mem_buf.get_int8buffer(embed_dim * embed_dim), 1, embed_dim, embed_dim);
    Matrix3D<int8_t> v_proj_bias(mem_buf.get_int8buffer(embed_dim), 1, 1, embed_dim);
    v_proj.weight = v_proj_weight;
    v_proj.bias = v_proj_bias;
    auto v_proj_op = W8A8B8O8Linear(v_proj);
    Matrix3D<int8_t> q_proj_weight(mem_buf.get_int8buffer(embed_dim * embed_dim), 1, embed_dim, embed_dim);
    Matrix3D<int8_t> q_proj_bias(mem_buf.get_int8buffer(embed_dim), 1, 1, embed_dim);
    q_proj.weight = q_proj_weight;
    q_proj.bias = q_proj_bias;
    auto q_proj_op = W8A8B8O8Linear(q_proj);

    struct W8A8BFP32OFP32Linear_params out_proj;
    Matrix3D<int8_t> out_proj_weight(mem_buf.get_int8buffer(embed_dim * embed_dim), 1, embed_dim, embed_dim);
    Matrix3D<float> out_proj_bias(mem_buf.get_fpbuffer(embed_dim), 1, 1, embed_dim);
    out_proj.weight = out_proj_weight;
    out_proj.bias = out_proj_bias;
    auto out_proj_op = W8A8BFP32OFP32Linear(out_proj);

    struct LayerNormQ_params self_attn_layer_norm, final_layer_norm;
    Matrix3D<float> self_attn_layer_norm_weight(mem_buf.get_fpbuffer(embed_dim), 1, 1, embed_dim);
    Matrix3D<float> self_attn_layer_norm_bias(mem_buf.get_fpbuffer(embed_dim), 1, 1, embed_dim);
    self_attn_layer_norm.weight = self_attn_layer_norm_weight;
    self_attn_layer_norm.bias = self_attn_layer_norm_bias;

    Matrix3D<float> final_layer_norm_weight(mem_buf.get_fpbuffer(embed_dim), 1, 1, embed_dim);
    Matrix3D<float> final_layer_norm_bias(mem_buf.get_fpbuffer(embed_dim), 1, 1, embed_dim);
    final_layer_norm.weight = final_layer_norm_weight;
    final_layer_norm.bias = final_layer_norm_bias;
    LayerNormQ self_attn_layer_norm_op = LayerNormQ(self_attn_layer_norm);
    LayerNormQ final_layer_norm_op = LayerNormQ(final_layer_norm);

    struct W8A8B8O8LinearReLU_params fc1;
    Matrix3D<int8_t> fc1_weight(mem_buf.get_int8buffer(embed_dim * hidden_dim), 1, hidden_dim, embed_dim);
    Matrix3D<int8_t> fc1_bias(mem_buf.get_int8buffer(hidden_dim), 1, 1, hidden_dim);
    fc1.weight = fc1_weight;
    fc1.bias_int8 = fc1_bias;
    auto fc1_op = W8A8B8O8LinearReLU(fc1);

    struct W8A8BFP32OFP32Linear_params fc2;
    Matrix3D<int8_t> fc2_weight(mem_buf.get_int8buffer(embed_dim * hidden_dim), 1, embed_dim, hidden_dim);
    Matrix3D<float> fc2_bias(mem_buf.get_fpbuffer(embed_dim), 1, 1, embed_dim);
    fc2.weight = fc2_weight;
    fc2.bias = fc2_bias;
    auto fc2_op = W8A8BFP32OFP32Linear(fc2);

    auto qk_bmm_op = BMM_S8T_S8N_F32T(qk_bmm);
    auto pv_bmm_op = BMM_S8T_S8N_S8T(pv_bmm);

    int layer_idx = 0;
    Int8OPTDecoderLayer layer = Int8OPTDecoderLayer(
        "models/OPT_125m/decoder/layer0", get_opt_model_config(OPT_125M), layer_idx, self_attn_layer_norm_op,
        final_layer_norm_op, fc1_op, fc2_op, qk_bmm_op, pv_bmm_op, k_proj_op, v_proj_op, q_proj_op, out_proj_op);

    Matrix3D<float> hidden_states(mem_buf.get_fpbuffer(b * sqlen * embed_dim), b, sqlen, embed_dim);
    read_to_array("assets/tests/OPT_125m/Decoder_layer_hidden_states.bin", hidden_states.m_data, b * sqlen * embed_dim);
    Matrix3D<float> attention_mask(mem_buf.get_fpbuffer(b * sqlen * sqlen), b, sqlen, sqlen);
    read_to_array("assets/tests/OPT_125m/Decoder_attention_mask.bin", attention_mask.m_data, b * sqlen * sqlen);
    struct Int8OPTDecoderLayer_input input = {hidden_states, attention_mask};

    struct Int8OPTDecoderLayer_output output = layer.forward(input);

    Matrix3D<float> residualGT(mem_buf.get_fpbuffer(b * sqlen * embed_dim), b, sqlen, embed_dim);
    read_to_array("assets/tests/OPT_125m/Decoder_residual.bin", residualGT.m_data, b * sqlen * embed_dim);
    int8_t *key_statesGT = mem_buf.get_int8buffer(output.past_key_value.first.length());
    read_to_array("assets/tests/OPT_125m/DecoderLayer_present_key.bin", key_statesGT,
                  output.past_key_value.first.length());
    int8_t *value_statesGT = mem_buf.get_int8buffer(output.past_key_value.second.length());
    read_to_array("assets/tests/OPT_125m/DecoderLayer_present_value.bin", value_statesGT,
                  output.past_key_value.second.length());

    bool sucess = check_two_equal(residualGT.m_data, output.hidden_states.m_data, b * sqlen * embed_dim, 4e-6);
    sucess &=
        check_two_equal(output.past_key_value.first.m_data, key_statesGT, output.past_key_value.first.length(), 6e-5);
    sucess &= check_two_equal(output.past_key_value.second.m_data, value_statesGT,
                              output.past_key_value.second.length(), 6e-5);
    // print_first_k_elelment("output.hidden_states.m_data", output.hidden_states.m_data, 10);
    // print_first_k_elelment("residualGT.m_data", residualGT.m_data, 10);
    if (!sucess)
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
