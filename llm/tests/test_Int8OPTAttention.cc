#include "Int8OPTAttention.h"
#include "operators.h"
#include "utils.h"
#include "utils_memalloc.h"

void test_Int8OPTAttention() {
    const int num_heads = 12, embed_dim = 768, sqlen = 108, b = 1;
    MemoryAllocator mem_buf;

    struct BMM_S8T_S8N_F32T_params qk_bmm;
    struct BMM_S8T_S8N_S8T_params pv_bmm;
    struct W8A8B8O8Linear_params k_proj, v_proj, q_proj;
    Matrix3D<int8_t> k_proj_weight(mem_buf.get_int8buffer(embed_dim * embed_dim), 1, embed_dim, embed_dim);
    Matrix3D<int8_t> k_proj_bias(mem_buf.get_int8buffer(embed_dim), 1, 1, embed_dim);
    k_proj.weight = k_proj_weight;
    k_proj.bias = k_proj_bias;
    auto k_proj_op = W8A8B8O8Linear(k_proj);
    // print_first_k_elelment("k_proj.weight", k_proj.weight.m_data, 10);

    Matrix3D<int8_t> v_proj_weight(mem_buf.get_int8buffer(embed_dim * embed_dim), 1, embed_dim, embed_dim);
    Matrix3D<int8_t> v_proj_bias(mem_buf.get_int8buffer(embed_dim), 1, 1, embed_dim);
    v_proj.weight = v_proj_weight;
    v_proj.bias = v_proj_bias;
    auto v_proj_op = W8A8B8O8Linear(v_proj);
    // print_first_k_elelment("v_proj.weight", v_proj.weight.m_data, 10);

    Matrix3D<int8_t> q_proj_weight(mem_buf.get_int8buffer(embed_dim * embed_dim), 1, embed_dim, embed_dim);
    Matrix3D<int8_t> q_proj_bias(mem_buf.get_int8buffer(embed_dim), 1, 1, embed_dim);
    q_proj.weight = q_proj_weight;
    q_proj.bias = q_proj_bias;
    auto q_proj_op = W8A8B8O8Linear(q_proj);
    // print_first_k_elelment("q_proj.weight", q_proj.weight.m_data, 10);

    struct W8A8BFP32OFP32Linear_params out_proj;
    Matrix3D<int8_t> out_proj_weight(mem_buf.get_int8buffer(embed_dim * embed_dim), 1, embed_dim, embed_dim);
    Matrix3D<float> out_proj_bias(mem_buf.get_fpbuffer(embed_dim), 1, 1, embed_dim);
    out_proj.weight = out_proj_weight;
    out_proj.bias = out_proj_bias;
    auto out_proj_op = W8A8BFP32OFP32Linear(out_proj);
    // print_first_k_elelment("out_proj.weight", out_proj.weight.m_data, 10);
    auto qk_bmm_op = BMM_S8T_S8N_F32T(qk_bmm);
    auto pv_bmm_op = BMM_S8T_S8N_S8T(pv_bmm);

    Int8OPTAttention::initialized_memory(get_opt_model_config(OPT_125M));
    Int8OPTAttention attn = Int8OPTAttention("models/OPT_125m/decoder/layer0/self_attn", get_opt_model_config(OPT_125M),
                                             qk_bmm_op, pv_bmm_op, k_proj_op, v_proj_op, q_proj_op, out_proj_op);

    Matrix3D<int8_t> hidden_states(mem_buf.get_int8buffer(embed_dim * sqlen), b, sqlen, embed_dim);
    hidden_states.load("assets/OPT/tests/attn/OPT_125m/Int8OPTAttention_hidden_states.bin");
    // print_first_k_elelment("hidden_states", hidden_states.m_data, 10);
    Matrix3D<float> attention_mask(mem_buf.get_fpbuffer(sqlen * sqlen), 1, sqlen, sqlen);
    attention_mask.load("assets/OPT/tests/attn/OPT_125m/Int8OPTAttention_attention_mask.bin");
    struct Int8OPTAttention_input input(hidden_states, attention_mask, 0);
    // print_first_k_elelment("input.hidden_states.m_data", input.hidden_states.m_data, 10);

    struct Int8OPTAttention_output output = attn.forward(input);

    Matrix3D<float> attn_outputGT(mem_buf.get_fpbuffer(b * sqlen * embed_dim), b, sqlen, embed_dim);
    attn_outputGT.load("assets/OPT/tests/attn/OPT_125m/Int8OPTAttention_attn_output.bin");
    Matrix3D<int8_t> key_statesGT(mem_buf.get_int8buffer(sqlen * embed_dim), num_heads, sqlen, embed_dim / num_heads);
    key_statesGT.load("assets/OPT/tests/attn/OPT_125m/Int8OPTAttention_key_states.bin");
    Matrix3D<int8_t> value_statesGT(mem_buf.get_int8buffer(sqlen * embed_dim), num_heads, sqlen, embed_dim / num_heads);
    value_statesGT.load("assets/OPT/tests/attn/OPT_125m/Int8OPTAttention_value_states.bin");

    bool success =
        check_two_exact_equal(value_statesGT.m_data, output.past_key_value.second.m_data, b * sqlen * embed_dim);
    success &= check_two_exact_equal(key_statesGT.m_data, output.past_key_value.first.m_data, b * sqlen * embed_dim);
    success &= check_two_equal(attn_outputGT.m_data, output.attn_output.m_data, b * sqlen * embed_dim);
    if (!success)
        std::cout << "Test of " << __func__ << ": Fail!" << std::endl;
    else
        std::cout << "-------- Test of " << __func__ << ": Passed! -------- " << std::endl;
}

void test_Int8OPTAttention_len512() {
    auto config = get_opt_model_config(OPT_125M);
    const int num_heads = config.num_heads, embed_dim = config.embed_dim, sqlen = 512, b = 1;
    MemoryAllocator mem_buf;

    struct BMM_S8T_S8N_F32T_params qk_bmm;
    struct BMM_S8T_S8N_S8T_params pv_bmm;
    struct W8A8B8O8Linear_params k_proj, v_proj, q_proj;
    Matrix3D<int8_t> k_proj_weight(mem_buf.get_int8buffer(embed_dim * embed_dim), 1, embed_dim, embed_dim);
    Matrix3D<int8_t> k_proj_bias(mem_buf.get_int8buffer(embed_dim), 1, 1, embed_dim);
    k_proj.weight = k_proj_weight;
    k_proj.bias = k_proj_bias;
    auto k_proj_op = W8A8B8O8Linear(k_proj);
    // print_first_k_elelment("k_proj.weight", k_proj.weight.m_data, 10);

    Matrix3D<int8_t> v_proj_weight(mem_buf.get_int8buffer(embed_dim * embed_dim), 1, embed_dim, embed_dim);
    Matrix3D<int8_t> v_proj_bias(mem_buf.get_int8buffer(embed_dim), 1, 1, embed_dim);
    v_proj.weight = v_proj_weight;
    v_proj.bias = v_proj_bias;
    auto v_proj_op = W8A8B8O8Linear(v_proj);
    // print_first_k_elelment("v_proj.weight", v_proj.weight.m_data, 10);

    Matrix3D<int8_t> q_proj_weight(mem_buf.get_int8buffer(embed_dim * embed_dim), 1, embed_dim, embed_dim);
    Matrix3D<int8_t> q_proj_bias(mem_buf.get_int8buffer(embed_dim), 1, 1, embed_dim);
    q_proj.weight = q_proj_weight;
    q_proj.bias = q_proj_bias;
    auto q_proj_op = W8A8B8O8Linear(q_proj);
    // print_first_k_elelment("q_proj.weight", q_proj.weight.m_data, 10);

    struct W8A8BFP32OFP32Linear_params out_proj;
    Matrix3D<int8_t> out_proj_weight(mem_buf.get_int8buffer(embed_dim * embed_dim), 1, embed_dim, embed_dim);
    Matrix3D<float> out_proj_bias(mem_buf.get_fpbuffer(embed_dim), 1, 1, embed_dim);
    out_proj.weight = out_proj_weight;
    out_proj.bias = out_proj_bias;
    auto out_proj_op = W8A8BFP32OFP32Linear(out_proj);
    // print_first_k_elelment("out_proj.weight", out_proj.weight.m_data, 10);
    auto qk_bmm_op = BMM_S8T_S8N_F32T(qk_bmm);
    auto pv_bmm_op = BMM_S8T_S8N_S8T(pv_bmm);

    Int8OPTAttention::initialized_memory(get_opt_model_config(OPT_125M));
    Int8OPTAttention attn = Int8OPTAttention("models/OPT_125m/decoder/layer0/self_attn", get_opt_model_config(OPT_125M),
                                             qk_bmm_op, pv_bmm_op, k_proj_op, v_proj_op, q_proj_op, out_proj_op);

    Matrix3D<int8_t> hidden_states(mem_buf.get_int8buffer(embed_dim * sqlen), b, sqlen, embed_dim);
    hidden_states.load("assets/OPT/tests/attn/OPT_125m/Int8OPTAttention_hidden_states_len512.bin");
    // print_first_k_elelment("hidden_states", hidden_states.m_data, 10);
    Matrix3D<float> attention_mask(mem_buf.get_fpbuffer(sqlen * sqlen), 1, sqlen, sqlen);
    attention_mask.load("assets/OPT/tests/attn/OPT_125m/Int8OPTAttention_attention_mask_len512.bin");
    struct Int8OPTAttention_input input(hidden_states, attention_mask, 0);
    // print_first_k_elelment("input.hidden_states.m_data", input.hidden_states.m_data, 10);

    struct Int8OPTAttention_output output = attn.forward(input);

    Matrix3D<float> attn_outputGT(mem_buf.get_fpbuffer(b * sqlen * embed_dim), b, sqlen, embed_dim);
    attn_outputGT.load("assets/OPT/tests/attn/OPT_125m/Int8OPTAttention_attn_output_len512.bin");
    Matrix3D<int8_t> key_statesGT(mem_buf.get_int8buffer(sqlen * embed_dim), num_heads, sqlen, embed_dim / num_heads);
    key_statesGT.load("assets/OPT/tests/attn/OPT_125m/Int8OPTAttention_key_states_len512.bin");
    Matrix3D<int8_t> value_statesGT(mem_buf.get_int8buffer(sqlen * embed_dim), num_heads, sqlen, embed_dim / num_heads);
    value_statesGT.load("assets/OPT/tests/attn/OPT_125m/Int8OPTAttention_value_states_len512.bin");

    bool success =
        check_two_exact_equal(value_statesGT.m_data, output.past_key_value.second.m_data, b * sqlen * embed_dim);
    success &= check_two_exact_equal(key_statesGT.m_data, output.past_key_value.first.m_data, b * sqlen * embed_dim);
    success &= check_two_equal(attn_outputGT.m_data, output.attn_output.m_data, b * sqlen * embed_dim);
    if (!success)
        std::cout << "Test of " << __func__ << ": Fail!" << std::endl;
    else
        std::cout << "-------- Test of " << __func__ << ": Passed! -------- " << std::endl;
}

void test_Int8OPTAttention_1_3B_len512() {
    auto config = get_opt_model_config(OPT_1_3B);
    const int num_heads = config.num_heads, embed_dim = config.embed_dim, sqlen = 512, b = 1;
    MemoryAllocator mem_buf;

    struct BMM_S8T_S8N_F32T_params qk_bmm;
    struct BMM_S8T_S8N_S8T_params pv_bmm;
    struct W8A8B8O8Linear_params k_proj, v_proj, q_proj;
    Matrix3D<int8_t> k_proj_weight(mem_buf.get_int8buffer(embed_dim * embed_dim), 1, embed_dim, embed_dim);
    Matrix3D<int8_t> k_proj_bias(mem_buf.get_int8buffer(embed_dim), 1, 1, embed_dim);
    k_proj.weight = k_proj_weight;
    k_proj.bias = k_proj_bias;
    auto k_proj_op = W8A8B8O8Linear(k_proj);
    // print_first_k_elelment("k_proj.weight", k_proj.weight.m_data, 10);

    Matrix3D<int8_t> v_proj_weight(mem_buf.get_int8buffer(embed_dim * embed_dim), 1, embed_dim, embed_dim);
    Matrix3D<int8_t> v_proj_bias(mem_buf.get_int8buffer(embed_dim), 1, 1, embed_dim);
    v_proj.weight = v_proj_weight;
    v_proj.bias = v_proj_bias;
    auto v_proj_op = W8A8B8O8Linear(v_proj);
    // print_first_k_elelment("v_proj.weight", v_proj.weight.m_data, 10);

    Matrix3D<int8_t> q_proj_weight(mem_buf.get_int8buffer(embed_dim * embed_dim), 1, embed_dim, embed_dim);
    Matrix3D<int8_t> q_proj_bias(mem_buf.get_int8buffer(embed_dim), 1, 1, embed_dim);
    q_proj.weight = q_proj_weight;
    q_proj.bias = q_proj_bias;
    auto q_proj_op = W8A8B8O8Linear(q_proj);
    // print_first_k_elelment("q_proj.weight", q_proj.weight.m_data, 10);

    struct W8A8BFP32OFP32Linear_params out_proj;
    Matrix3D<int8_t> out_proj_weight(mem_buf.get_int8buffer(embed_dim * embed_dim), 1, embed_dim, embed_dim);
    Matrix3D<float> out_proj_bias(mem_buf.get_fpbuffer(embed_dim), 1, 1, embed_dim);
    out_proj.weight = out_proj_weight;
    out_proj.bias = out_proj_bias;
    auto out_proj_op = W8A8BFP32OFP32Linear(out_proj);
    // print_first_k_elelment("out_proj.weight", out_proj.weight.m_data, 10);
    auto qk_bmm_op = BMM_S8T_S8N_F32T(qk_bmm);
    auto pv_bmm_op = BMM_S8T_S8N_S8T(pv_bmm);

    Int8OPTAttention::initialized_memory(get_opt_model_config(OPT_1_3B));
    Int8OPTAttention attn = Int8OPTAttention("models/OPT_1.3B/decoder/layer0/self_attn", get_opt_model_config(OPT_1_3B),
                                             qk_bmm_op, pv_bmm_op, k_proj_op, v_proj_op, q_proj_op, out_proj_op);

    Matrix3D<int8_t> hidden_states(mem_buf.get_int8buffer(embed_dim * sqlen), b, sqlen, embed_dim);
    hidden_states.load("assets/OPT/tests/attn/OPT_1.3B/Int8OPTAttention_hidden_states_len512.bin");
    // print_first_k_elelment("hidden_states", hidden_states.m_data, 10);
    Matrix3D<float> attention_mask(mem_buf.get_fpbuffer(sqlen * sqlen), 1, sqlen, sqlen);
    attention_mask.load("assets/OPT/tests/attn/OPT_1.3B/Int8OPTAttention_attention_mask_len512.bin");
    struct Int8OPTAttention_input input(hidden_states, attention_mask, 0);
    // print_first_k_elelment("input.hidden_states.m_data", input.hidden_states.m_data, 10);

    struct Int8OPTAttention_output output = attn.forward(input);

    Matrix3D<float> attn_outputGT(mem_buf.get_fpbuffer(b * sqlen * embed_dim), b, sqlen, embed_dim);
    attn_outputGT.load("assets/OPT/tests/attn/OPT_1.3B/Int8OPTAttention_attn_output_len512.bin");
    Matrix3D<int8_t> key_statesGT(mem_buf.get_int8buffer(sqlen * embed_dim), num_heads, sqlen, embed_dim / num_heads);
    key_statesGT.load("assets/OPT/tests/attn/OPT_1.3B/Int8OPTAttention_key_states_len512.bin");
    Matrix3D<int8_t> value_statesGT(mem_buf.get_int8buffer(sqlen * embed_dim), num_heads, sqlen, embed_dim / num_heads);
    value_statesGT.load("assets/OPT/tests/attn/OPT_1.3B/Int8OPTAttention_value_states_len512.bin");

    bool success =
        check_two_exact_equal(value_statesGT.m_data, output.past_key_value.second.m_data, b * sqlen * embed_dim);
    success &= check_two_exact_equal(key_statesGT.m_data, output.past_key_value.first.m_data, b * sqlen * embed_dim);
    success &= check_two_equal(attn_outputGT.m_data, output.attn_output.m_data, b * sqlen * embed_dim, 1e-7);
    if (!success)
        std::cout << "Test of " << __func__ << ": Fail!" << std::endl;
    else
        std::cout << "-------- Test of " << __func__ << ": Passed! -------- " << std::endl;
}

int main() {
    test_Int8OPTAttention();
    test_Int8OPTAttention_len512();
    test_Int8OPTAttention_1_3B_len512();
}
