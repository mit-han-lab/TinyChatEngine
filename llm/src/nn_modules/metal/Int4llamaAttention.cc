#include <string.h>
#include <cmath>
#include <cfloat>

#include "Int4llamaAttention.h"
#include "operators.h"
#include "utils.h"
#include "metal_compute.h"

static float16_t *attn_weights_arr = nullptr;
static float16_t *attn_output_half_arr = nullptr;
static float16_t *attn_output_arr = nullptr;
static float16_t *attn_output_transpose_arr = nullptr;
static float16_t *key_states_arr = nullptr;
static float16_t *value_states_arr = nullptr;
static float16_t *query_states_arr = nullptr;
static float16_t *value_states_transpose_arr = nullptr;
static float16_t *key_states_arr_cache = nullptr;
static float16_t *value_states_arr_cache = nullptr;
static int *cache_num = nullptr;
static float16_t *qkv_states_unshape_arr = nullptr;

void Int4llamaAttention::initialized_memory(const struct model_config config) {
    allocate_aligned_memory(attn_weights_arr, config.num_heads * config.max_sqlen * config.max_sqlen * sizeof(float16_t));
    allocate_aligned_memory(attn_output_half_arr, config.max_sqlen * config.embed_dim * sizeof(float16_t));
    allocate_aligned_memory(attn_output_arr, config.max_sqlen * config.embed_dim * sizeof(float16_t));
    allocate_aligned_memory(attn_output_transpose_arr, config.max_sqlen * config.embed_dim * sizeof(float16_t));
    allocate_aligned_memory(key_states_arr, config.max_sqlen * config.embed_dim * sizeof(float16_t));
    allocate_aligned_memory(value_states_arr, config.max_sqlen * config.embed_dim * sizeof(float16_t));
    allocate_aligned_memory(query_states_arr, config.max_sqlen * config.embed_dim * sizeof(float16_t));
    allocate_aligned_memory(value_states_transpose_arr, config.max_sqlen * config.embed_dim * sizeof(float16_t));

    allocate_aligned_memory(cache_num, config.num_layers * sizeof(int));
    for (int i = 0; i < config.num_layers; i++) cache_num[i] = 0;

    allocate_aligned_memory(key_states_arr_cache, config.num_layers * 2 * config.max_sqlen * config.embed_dim * sizeof(float16_t));
    allocate_aligned_memory(value_states_arr_cache, config.num_layers * 2 * config.max_sqlen * config.embed_dim * sizeof(float16_t));
    allocate_aligned_memory(qkv_states_unshape_arr, config.max_sqlen * config.embed_dim * 3 * sizeof(float16_t));
}

Int4llamaAttention::Int4llamaAttention(std::string param_path, const struct model_config config, int layer_idx) {
    allocate_aligned_memory(o_weight, (config.embed_dim * config.embed_dim * sizeof(int)) / 8);
    allocate_aligned_memory(qkv_weight, (config.embed_dim * config.embed_dim * 3 * sizeof(int)) / 8);
    this->o_proj = Linear_half_int4(Matrix3D<int>(o_weight, 1, config.embed_dim, config.embed_dim / 8),
                                  param_path + "/o_proj");
    this->qkv_proj = Linear_half_int4(Matrix3D<int>(qkv_weight, 1, config.embed_dim, config.embed_dim * 3 / 8), 
                                  param_path + "/qkv_proj");

    allocate_aligned_memory(cos_buf, config.max_sqlen * (config.embed_dim / config.num_heads) * sizeof(half));
    allocate_aligned_memory(sin_buf, config.max_sqlen * (config.embed_dim / config.num_heads) * sizeof(half));
    Matrix3D<half> cos(cos_buf, 1, config.max_sqlen, (config.embed_dim / config.num_heads));
    Matrix3D<half> sin(sin_buf, 1, config.max_sqlen, (config.embed_dim / config.num_heads));

    this->rotary_pos_emb = RotaryPosEmb_metal(cos, sin, param_path + "/rotary_emb");

    half qk_bmm_alpha;
    read_to_array_half((param_path + "/qk_bmm/alpha_half.bin").c_str(), &qk_bmm_alpha, 1);
    this->qk_bmm = BMM_F16T(qk_bmm_alpha);
    this->pv_bmm = BMM_F16T((half)(1.0f)); //float2half?

    this->embed_dim = config.embed_dim;
    this->num_heads = config.num_heads;
    assert(config.embed_dim % config.num_heads == 0);
    this->head_dim = config.embed_dim / config.num_heads;
    this->max_sqlen = config.max_sqlen;
}

void shape_qkv(Matrix3D<float16_t> qkv_states_unshape, Matrix3D<float16_t> query_states, Matrix3D<float16_t> key_states, Matrix3D<float16_t> value_states, int num_heads, int sqlen, int head_dim){
    struct metal_params params;

    params.A.half_data_ptr = qkv_states_unshape.m_data;
    params.B.half_data_ptr = query_states.m_data;
    params.C.half_data_ptr = key_states.m_data;
    params.D.half_data_ptr = value_states.m_data;
    params.num_heads = num_heads;
    params.sqlen = sqlen;
    params.head_dim = head_dim;
    params.op = METAL_KERNEL_SHAPE_QKV;
    add_node(&params);
}

void unshape(Matrix3D<float16_t> attn_output, Matrix3D<float16_t> attn_output_transpose, int num_heads, int sqlen, int head_dim){
    struct metal_params params;

    params.A.half_data_ptr = attn_output.m_data;
    params.B.half_data_ptr = attn_output_transpose.m_data;
    params.num_heads = num_heads;
    params.sqlen = sqlen;
    params.head_dim = head_dim;
    params.op = METAL_KERNEL_UNSHAPE;
    add_node(&params);
}

void check_inf_half(Matrix3D<float16_t> attn_weights){
    struct metal_params params;

    params.A.half_data_ptr = attn_weights.m_data;
    params.sqlen = attn_weights.length();
    params.op = METAL_KERNEL_CHECK_INF_HALF;
    add_node(&params);
    return;
}

void transpose_1_2idx(Matrix3D<float16_t> final_value_states, Matrix3D<float16_t> value_states_transpose, int num_heads, int sqlen, int head_dim, int tgz){
    struct metal_params params;

    params.A.half_data_ptr = final_value_states.m_data;
    params.A.row = final_value_states.m_dim_x;
    params.A.column = final_value_states.m_dim_y;
    params.input_m_dim_z = final_value_states.m_dim_z;
    params.B.half_data_ptr = value_states_transpose.m_data;
    params.B.row = value_states_transpose.m_dim_x;
    params.B.column = value_states_transpose.m_dim_y;
    params.num_heads = num_heads;
    params.sqlen = sqlen;
    params.head_dim = head_dim;
    params.tgz = tgz;
    params.op = METAL_KERNEL_TRANSPOSE_1_2IDX;
    add_node(&params);
    return;
}

struct Int4llamaAttention_output Int4llamaAttention::forward(std::string param_path, const struct Int4llamaAttention_input &input) {
    PROFILE_START(profile_name);

    struct Int4llamaAttention_output output;
    const int sqlen = input.hidden_states.m_dim_y, b = input.hidden_states.m_dim_x;
    assert(b == 1);

    // Fused QKV
    Matrix3D<float16_t> qkv_states_unshape(qkv_states_unshape_arr, b, sqlen, embed_dim * 3);
    this->qkv_proj.forward(input.hidden_states, qkv_states_unshape);
    Matrix3D<float16_t> query_states(query_states_arr, this->num_heads, sqlen, this->head_dim);
    Matrix3D<float16_t> key_states(key_states_arr, this->num_heads, sqlen, this->head_dim);
    Matrix3D<float16_t> value_states(value_states_arr, this->num_heads, sqlen, this->head_dim);
    // METAL: more kernels needed
    shape_qkv(qkv_states_unshape, query_states, key_states, value_states, this->num_heads, sqlen, this->head_dim);

    int tgz = sqlen;
    if (input.has_past_key_value) {
        assert(input.past_key.m_dim_z == this->head_dim);
        tgz += input.past_key.m_dim_y;    
    }
    float16_t *ret_value_states, *ret_key_states;
    if (cache_num[input.layer_idx] == 1) {
        ret_value_states = &value_states_arr_cache[(input.layer_idx * 2 + 1) * this->max_sqlen * this->embed_dim];
        ret_key_states = &key_states_arr_cache[(input.layer_idx * 2 + 1) * this->max_sqlen * this->embed_dim];
        cache_num[input.layer_idx] = 0;
    } else {
        ret_value_states = &value_states_arr_cache[input.layer_idx * 2 * this->max_sqlen * this->embed_dim];
        ret_key_states = &key_states_arr_cache[input.layer_idx * 2 * this->max_sqlen * this->embed_dim];
        cache_num[input.layer_idx] = 1;
    }
    Matrix3D<float16_t> final_value_states(ret_value_states, this->num_heads, tgz, this->head_dim);
    Matrix3D<float16_t> final_key_states(ret_key_states, this->num_heads, tgz, this->head_dim);
    Matrix3D<float16_t> attn_output_half(attn_output_half_arr, 1, sqlen, this->num_heads * this->head_dim);

    int start_idx = 0;
    if (input.has_past_key_value) start_idx = input.past_key.m_dim_y;

    // dim3 grid(num_heads, 1, 1);
    // dim3 block(sqlen, 1, 1);
    // METAL: ROPE metal
    RotaryPosEmb_metal_forward(query_states, key_states, this->rotary_pos_emb.cos, this->rotary_pos_emb.sin, start_idx, sqlen);

    // int tgz = sqlen;
//     if (input.has_past_key_value) {
//         // assert(input.past_key.m_dim_z == this->head_dim);
//         // tgz += input.past_key.m_dim_y;
//         float16_t *val_ptr = ret_value_states, *key_ptr = ret_key_states;
//         int past_block = input.past_key.m_dim_y * input.past_key.m_dim_z;
//         int sq_block = sqlen * this->head_dim;
// #pragma unroll
//         for (int i = 0; i < input.past_key.m_dim_x; i++) {
//             cudaMemcpyAsync(val_ptr, &input.past_value.m_data[past_block * i], past_block * sizeof(float16_t), cudaMemcpyDeviceToDevice);
//             val_ptr += past_block;
//             cudaMemcpyAsync(val_ptr, &value_states.m_data[sq_block * i], sq_block * sizeof(float16_t), cudaMemcpyDeviceToDevice);
//             val_ptr += sq_block;
//             cudaMemcpyAsync(key_ptr, &input.past_key.m_data[past_block * i], past_block * sizeof(float16_t), cudaMemcpyDeviceToDevice);
//             key_ptr += past_block;
//             cudaMemcpyAsync(key_ptr, &key_states.m_data[sq_block * i], sq_block * sizeof(float16_t), cudaMemcpyDeviceToDevice);
//             key_ptr += sq_block;
//         }
//     } else {
//         cudaMemcpyAsync(ret_value_states, value_states_arr, (this->num_heads * tgz * this->head_dim) * sizeof(float16_t), cudaMemcpyDeviceToDevice);
//         cudaMemcpyAsync(ret_key_states, key_states_arr, (this->num_heads * tgz * this->head_dim) * sizeof(float16_t), cudaMemcpyDeviceToDevice);
//     }

    Matrix3D<float16_t> attn_weights(attn_weights_arr, this->num_heads, sqlen, tgz);
    this->qk_bmm.forward(query_states, final_key_states, attn_weights);

    // dim3 threadsPerBlock2(16, 4, 16);
    // dim3 numBlocks2((this->num_heads + threadsPerBlock2.x - 1) / threadsPerBlock2.x,
    //             (sqlen + threadsPerBlock2.y - 1) / threadsPerBlock2.y,
    //             (tgz + threadsPerBlock2.z - 1) / threadsPerBlock2.z);
    // METAL: Metal
    batch_Add_metal(attn_weights, input.attention_mask, attn_weights);

    int threadsPerBlock_1D = 1024;
    int blocksPerGrid_1D =(attn_weights.length() + threadsPerBlock_1D - 1) / threadsPerBlock_1D;
    // METAL: more kernels needed
    check_inf_half(attn_weights);

    Matrix3D<float16_t> attn_probs(attn_weights_arr, this->num_heads, sqlen, tgz);
    // dim3 threadsPerBlock3(64, 16);
    // dim3 numBlocks3((this->num_heads + threadsPerBlock3.x - 1) / threadsPerBlock3.x, (sqlen + threadsPerBlock3.y - 1) / threadsPerBlock3.y);
    // METAL: Metal
    softmax(attn_weights, attn_probs);

    /* Legacy Implementation of PV_BMM*/
    Matrix3D<float16_t> value_states_transpose(value_states_transpose_arr, this->num_heads, this->head_dim, tgz);
    // METAL: more kernels needed
    transpose_1_2idx(final_value_states, value_states_transpose, this->num_heads, sqlen, this->head_dim, tgz);

    Matrix3D<float16_t> attn_output(attn_output_arr, this->num_heads, sqlen, this->head_dim);
    this->pv_bmm.forward(attn_probs, value_states_transpose, attn_output);
    /* Alternative Implementation (untransposed) of PV_BMM*/
    // Matrix3D<float16_t> attn_output(attn_output_arr, this->num_heads, sqlen, this->head_dim);
    // this->pv_bmm.forward_weight_untransposed(attn_probs, final_value_states, attn_output);

    Matrix3D<float16_t> attn_output_transpose(attn_output_transpose_arr, 1, sqlen, this->num_heads * this->head_dim);
    // METAL: more kernels
    unshape(attn_output, attn_output_transpose, this->num_heads, sqlen, this->head_dim);

    // Matrix3D<float16_t> attn_output_half(attn_output_half_arr, 1, sqlen, this->num_heads * this->head_dim);
    this->o_proj.forward(attn_output_transpose, attn_output_half);

    // output assignment
    output.attn_output = attn_output_half;
    output.past_key_value = {final_key_states, final_value_states};

    PROFILE_END(profile_name);

    return output;
}