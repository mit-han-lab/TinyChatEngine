#include "Fp32GPTBigCodeAttention.h"

#include <string.h>

#include <cmath>

#include "operators.h"
#include "utils.h"

static float *attn_weights_arr;
static float *attn_probs_arr;
// static float *attn_probs_int8_arr;
static float ***key_states_arr_cache;
static float ***value_states_arr_cache;
static float *attn_output_fp_arr;
static int *cache_num;
static float *attn_output_arr;
static float *attn_output_transpose_arr;
static float *key_states_arr;
static float *value_states_arr;
static float *query_states_arr;
static float *value_states_transpose_arr;
// static float *query_states_unshape_arr;
// static float *key_states_unshape_arr;
// static float *value_states_unshape_arr;
static float *qkv_states_unshape_arr;

void Fp32GPTBigCodeAttention::initialized_memory(const struct model_config config) {
    allocate_aligned_memory(attn_weights_arr, config.num_heads * config.max_sqlen * config.max_sqlen * sizeof(float));
    allocate_aligned_memory(attn_probs_arr, config.num_heads * config.max_sqlen * config.max_sqlen * sizeof(float));
    // allocate_aligned_memory(attn_probs_int8_arr,
    //                         config.num_heads * config.max_sqlen * config.max_sqlen * sizeof(float));
    allocate_aligned_memory(attn_output_fp_arr, config.max_sqlen * config.embed_dim * sizeof(float));
    allocate_aligned_memory(attn_output_arr, config.max_sqlen * config.embed_dim * sizeof(float));
    allocate_aligned_memory(attn_output_transpose_arr, config.max_sqlen * config.embed_dim * sizeof(float));
    allocate_aligned_memory(key_states_arr, config.max_sqlen * config.embed_dim * sizeof(float));
    allocate_aligned_memory(value_states_arr, config.max_sqlen * config.embed_dim * sizeof(float));
    allocate_aligned_memory(query_states_arr, config.max_sqlen * config.embed_dim * sizeof(float));
    allocate_aligned_memory(value_states_transpose_arr, config.max_sqlen * config.embed_dim * sizeof(float));
    cache_num = new int[config.num_layers];
    for (int i = 0; i < config.num_layers; i++) cache_num[i] = 0;
    allocate_aligned_memory(qkv_states_unshape_arr, config.max_sqlen * (config.embed_dim + 2 * config.embed_dim / config.num_heads) * sizeof(float));
    key_states_arr_cache = new float **[config.num_layers];
    for (int i = 0; i < config.num_layers; ++i) {
        key_states_arr_cache[i] = new float *[2];
        for (int j = 0; j < 2; ++j) {
            allocate_aligned_memory(key_states_arr_cache[i][j], config.max_sqlen * config.embed_dim * sizeof(float));
        }
    }
    value_states_arr_cache = new float **[config.num_layers];
    for (int i = 0; i < config.num_layers; ++i) {
        value_states_arr_cache[i] = new float *[2];
        for (int j = 0; j < 2; ++j) {
            allocate_aligned_memory(value_states_arr_cache[i][j], config.max_sqlen * config.embed_dim * sizeof(float));
        }
    }

    // allocate_aligned_memory(query_states_unshape_arr, config.max_sqlen * config.embed_dim * sizeof(float));
    // allocate_aligned_memory(key_states_unshape_arr, config.max_sqlen * config.embed_dim * sizeof(float));
    // allocate_aligned_memory(value_states_unshape_arr, config.max_sqlen * config.embed_dim * sizeof(float));
    allocate_aligned_memory(qkv_states_unshape_arr, config.max_sqlen * config.embed_dim * 3 * sizeof(float));
}

Fp32GPTBigCodeAttention::Fp32GPTBigCodeAttention(std::string param_path, const struct model_config config) {
    this->embed_dim = config.embed_dim;
    this->num_heads = config.num_heads;
    this->head_dim = config.embed_dim / config.num_heads;
    this->kv_heads = 1;
    this->kv_dim = this->kv_heads * this->head_dim;
    assert(config.embed_dim % config.num_heads == 0);

    float *c_attn_weight, *c_proj_weight;
    // allocate_aligned_memory(c_attn_weight, (config.embed_dim * (config.embed_dim + 2 * this->kv_dim) * sizeof(float)));
    allocate_aligned_memory(c_attn_weight, (config.embed_dim * config.embed_dim * 3 * sizeof(float)));
    allocate_aligned_memory(c_proj_weight, (config.embed_dim * config.embed_dim * sizeof(float)));
    float *c_attn_bias, *c_proj_bias;
    // allocate_aligned_memory(c_attn_bias, ((config.embed_dim + 2 * this->kv_dim) * sizeof(float)));
    allocate_aligned_memory(c_attn_bias, (config.embed_dim * 3 * sizeof(float)));
    allocate_aligned_memory(c_proj_bias, (config.embed_dim * sizeof(float)));

    // this->c_attn =
    //     Linear_FP(Matrix3D<float>(c_attn_weight, 1, (config.embed_dim + 2 * this->kv_dim), config.embed_dim), param_path + "/c_attn/weight.bin",
    //               Matrix3D<float>(c_attn_bias, 1, 1, (config.embed_dim + 2 * this->kv_dim)), param_path + "/c_attn/bias.bin");
    this->c_attn =
        Linear_FP(Matrix3D<float>(c_attn_weight, 1, (config.embed_dim * 3), config.embed_dim), param_path + "/c_attn/weight.bin",
                  Matrix3D<float>(c_attn_bias, 1, 1, (config.embed_dim * 3)), param_path + "/c_attn/bias.bin");
    this->c_attn.has_bias = true;

    this->c_proj =
        Linear_FP(Matrix3D<float>(c_proj_weight, 1, config.embed_dim, config.embed_dim), param_path + "/c_proj/weight.bin",
                  Matrix3D<float>(c_proj_bias, 1, 1, config.embed_dim), param_path + "/c_proj/bias.bin");
    this->c_proj.has_bias = true;

    float qk_bmm_alpha;
    read_to_array((param_path + "/qk_bmm/alpha.bin").c_str(), &qk_bmm_alpha, 1);
    this->qk_bmm = BMM_F32T(qk_bmm_alpha);
    this->pv_bmm = BMM_F32T(1.0f);
}

inline void Fp32GPTBigCodeAttention::shape_qkv(Matrix3D<float> unshape, Matrix3D<float> shaped_q, Matrix3D<float> shaped_k,
                                          Matrix3D<float> shaped_v, int sqlen) {
    PROFILE_START("Fp32GPTBigCodeAttention::shape_qkv");
    assert(unshape.m_dim_x == 1);  // bsz == 1
    assert(unshape.m_dim_y == sqlen);
    assert(unshape.m_dim_z == this->num_heads * this->head_dim * 3);
    assert(shaped_q.m_dim_x == this->num_heads);
    assert(shaped_q.m_dim_y == sqlen);
    assert(shaped_q.m_dim_z == this->head_dim);
    assert(shaped_k.m_dim_x == this->num_heads);
    assert(shaped_k.m_dim_y == sqlen);
    assert(shaped_k.m_dim_z == this->head_dim);
    assert(shaped_v.m_dim_x == this->num_heads);
    assert(shaped_v.m_dim_y == sqlen);
    assert(shaped_v.m_dim_z == this->head_dim);

    for (int i = 0; i < this->num_heads; i++) {
        for (int j = 0; j < sqlen; j++) {
            for (int k = 0; k < this->head_dim; k++) {
                shaped_q(i, j, k) = unshape(0, j, i * this->head_dim + k);
            }
            for (int k = this->embed_dim; k < this->embed_dim + this->head_dim; k++) {
                shaped_k(i, j, k - this->embed_dim) = unshape(0, j, i * this->head_dim + k);
            }
            for (int k = this->embed_dim * 2; k < this->embed_dim * 2 + this->head_dim; k++) {
                shaped_v(i, j, k - this->embed_dim * 2) = unshape(0, j, i * this->head_dim + k);
            }
        }
    }

    PROFILE_END("Fp32GPTBigCodeAttention::shape_qkv");
}

inline void Fp32GPTBigCodeAttention::unshape(Matrix3D<float> shaped, Matrix3D<float> unshape, int sqlen) {
    PROFILE_START("Fp32GPTBigCodeAttention::unshpae");
    assert(unshape.m_dim_x == 1);  // bsz == 1
    assert(unshape.m_dim_y == sqlen);
    assert(unshape.m_dim_z == this->num_heads * this->head_dim);
    assert(shaped.m_dim_x == this->num_heads);
    assert(shaped.m_dim_y == sqlen);
    assert(shaped.m_dim_z == this->head_dim);

    for (int i = 0; i < this->num_heads; i++) {
        for (int j = 0; j < sqlen; j++) {
            for (int k = 0; k < this->head_dim; k++) {
                unshape(0, j, i * this->head_dim + k) = shaped(i, j, k);
            }
        }
    }
    PROFILE_END("Fp32GPTBigCodeAttention::unshpae");
}

struct transpose_1_2idx_arg {
    int start_idx, end_idx;
    Matrix3D<float> input, output;
};

template <typename T>
inline void transpose_1_2idx(Matrix3D<T> &input, Matrix3D<T> &output) {
    PROFILE_START("Fp32GPTBigCodeAttention::transpose_1_2idx");
    assert(input.m_dim_x == output.m_dim_x);
    assert(input.m_dim_y == output.m_dim_z);
    assert(input.m_dim_z == output.m_dim_y);

    for (int i = 0; i < input.m_dim_x; i++) {
        for (int j = 0; j < input.m_dim_y; j++) {
            for (int k = 0; k < input.m_dim_z; k++) {
                output(i, k, j) = input(i, j, k);
            }
        }
    }
    PROFILE_END("Fp32GPTBigCodeAttention::transpose_1_2idx");
}

struct Fp32GPTBigCodeAttention_output Fp32GPTBigCodeAttention::forward(const struct Fp32GPTBigCodeAttention_input &input) {
    PROFILE_START(profile_name);
    struct Fp32GPTBigCodeAttention_output output;
    const int sqlen = input.hidden_states.m_dim_y, b = input.hidden_states.m_dim_x;
    assert(b == 1);

    // Fused QKV
    Matrix3D<float> qkv_states_unshape(qkv_states_unshape_arr, b, sqlen, embed_dim * 3);
    this->c_attn.forward(input.hidden_states, qkv_states_unshape);
    Matrix3D<float> query_states(query_states_arr, this->num_heads, sqlen, this->head_dim);
    Matrix3D<float> key_states(key_states_arr, this->num_heads, sqlen, this->head_dim);
    Matrix3D<float> value_states(value_states_arr, this->num_heads, sqlen, this->head_dim);
    this->shape_qkv(qkv_states_unshape, query_states, key_states, value_states, sqlen);
    
    // // Query, Key and Value states
    // Matrix3D<float> qkv_states_unshape(qkv_states_unshape_arr, b, sqlen, this->embed_dim + 2 * this->kv_dim);
    // this->c_attn.forward(input.hidden_states, qkv_states_unshape);
    // // for (int i = 0; i < sqlen * embed_dim; i++) qkv_states_unshape.m_data[i] *= this->scaling;  // To be checked
    // Matrix3D<float> key_states(key_states_arr, this->kv_heads, sqlen, this->head_dim);
    // Matrix3D<float> value_states(value_states_arr, this->kv_heads, sqlen, this->head_dim);
    // assert(!std::isnan(qkv_states_unshape_arr[0]));

    // // Reshape
    // Matrix3D<float> query_states(query_states_arr, this->num_heads, sqlen, this->head_dim);
    // this->shape_MQA_qkv(qkv_states_unshape, query_states, sqlen);

    // Get the memory buffer
    float *ret_value_states, *ret_key_states;
    if (cache_num[input.layer_idx] == 1) {
        ret_value_states = value_states_arr_cache[input.layer_idx][1];
        ret_key_states = key_states_arr_cache[input.layer_idx][1];
        cache_num[input.layer_idx] = 0;
    } else {
        ret_value_states = value_states_arr_cache[input.layer_idx][0];
        ret_key_states = key_states_arr_cache[input.layer_idx][0];
        cache_num[input.layer_idx] = 1;
    }

    // // Key states
    // Matrix3D<float> key_states_unshape(key_states_unshape_arr, b, sqlen, embed_dim);
    // this->k_proj.forward(input.hidden_states, key_states_unshape);
    // Matrix3D<float> key_states(key_states_arr, this->num_heads, sqlen, this->head_dim);
    // this->shpae(key_states_unshape, key_states, sqlen);

    // assert(!std::isnan(key_states_unshape.sum()));

    // // Value states
    // Matrix3D<float> value_states_unshape(value_states_unshape_arr, b, sqlen, embed_dim);
    // this->v_proj.forward(input.hidden_states, value_states_unshape);
    // Matrix3D<float> value_states(value_states_arr, this->num_heads, sqlen, this->head_dim);
    // this->shpae(value_states_unshape, value_states, sqlen);

    // assert(!std::isnan(value_states_unshape_arr[0]));

    // Concate with past key, value if exists
    PROFILE_START(profile_name + "::cat_past_keys_values");
    int tgz = sqlen;
    if (input.has_past_key_value) {
        assert(input.past_key.m_dim_z == this->head_dim);
        tgz += input.past_key.m_dim_y;
        float *val_ptr = ret_value_states, *key_ptr = ret_key_states;
        int past_block = input.past_key.m_dim_y * input.past_key.m_dim_z;
        int sq_block = sqlen * this->head_dim;
        for (int i = 0; i < input.past_key.m_dim_x; i++) {
            memcpy(val_ptr, &input.past_value.m_data[past_block * i], past_block * sizeof(float));
            val_ptr += past_block;
            memcpy(val_ptr, &value_states.m_data[sq_block * i], sq_block * sizeof(float));
            val_ptr += sq_block;
            memcpy(key_ptr, &input.past_key.m_data[past_block * i], past_block * sizeof(float));
            key_ptr += past_block;
            memcpy(key_ptr, &key_states.m_data[sq_block * i], sq_block * sizeof(float));
            key_ptr += sq_block;
        }
    } else {
        // Put the data into the buffer
        memcpy(ret_value_states, value_states_arr, (this->num_heads * tgz * this->head_dim) * sizeof(float));
        memcpy(ret_key_states, key_states_arr, (this->num_heads * tgz * this->head_dim) * sizeof(float));
    }
    Matrix3D<float> final_value_states(ret_value_states, this->num_heads, tgz, this->head_dim);
    Matrix3D<float> final_key_states(ret_key_states, this->num_heads, tgz, this->head_dim);
    PROFILE_END(profile_name + "::cat_past_keys_values");

    // QK_BMM
    Matrix3D<float> attn_weights(attn_weights_arr, this->num_heads, sqlen, tgz);
    this->qk_bmm.forward(query_states, final_key_states, attn_weights);

    // Add mask
    batch_Add(attn_weights, input.attention_mask, attn_weights);

    // Softmax QK
    Matrix3D<float> attn_probs(attn_weights_arr, this->num_heads, sqlen, tgz);
    softmax(attn_weights, attn_probs, 2);

    // Transpose V for PV_BMM
    Matrix3D<float> value_states_transpose(value_states_transpose_arr, this->num_heads, this->head_dim, tgz);
    transpose_1_2idx(final_value_states, value_states_transpose);
    Matrix3D<float> attn_output(attn_output_arr, this->num_heads, sqlen, this->head_dim);
    this->pv_bmm.forward(attn_probs, value_states_transpose, attn_output);

    Matrix3D<float> attn_output_transpose(attn_output_transpose_arr, 1, sqlen, this->num_heads * this->head_dim);
    this->unshape(attn_output, attn_output_transpose, sqlen);

    // Output projection
    Matrix3D<float> attn_output_fp(attn_output_fp_arr, 1, sqlen, this->num_heads * this->head_dim);
    this->c_proj.forward(attn_output_transpose, attn_output_fp);

    // Output assignment
    output.attn_output = attn_output_fp;
    output.past_key_value = {final_key_states, final_value_states};

    PROFILE_END(profile_name);
    return output;
}
