#include "int4llamaAttention.h"

#include <string.h>

#include <cmath>

#include "llama_utils.h"
#include "operators.h"
#include "utils.h"

static float *attn_weights_arr;
static float ***key_states_arr_cache;
static float ***value_states_arr_cache;
static float *attn_output_fp_arr;
static int *cache_num;
static float *query_states_unshape_arr;
static float *attn_output_arr;
static float *attn_output_transpose_arr;
static float *key_states_unshape_arr;
static float *key_states_arr;
static float *value_states_unshape_arr;
static float *value_states_arr;
static float *query_states_arr;
static float *value_states_transpose_arr;

void int4llamaAttention::initialized_memory(const struct model_config config) {
    allocate_aligned_memory(attn_weights_arr, config.num_heads * config.max_sqlen * config.max_sqlen * sizeof(float));
    allocate_aligned_memory(attn_output_fp_arr, config.max_sqlen * config.embed_dim * sizeof(float));
    allocate_aligned_memory(attn_output_arr, config.max_sqlen * config.embed_dim * sizeof(float));
    allocate_aligned_memory(attn_output_transpose_arr, config.max_sqlen * config.embed_dim * sizeof(float));
    allocate_aligned_memory(key_states_unshape_arr, config.max_sqlen * config.embed_dim * sizeof(float));
    allocate_aligned_memory(key_states_arr, config.max_sqlen * config.embed_dim * sizeof(float));
    allocate_aligned_memory(value_states_unshape_arr, config.max_sqlen * config.embed_dim * sizeof(float));
    allocate_aligned_memory(value_states_arr, config.max_sqlen * config.embed_dim * sizeof(float));
    allocate_aligned_memory(query_states_arr, config.max_sqlen * config.embed_dim * sizeof(float));
    allocate_aligned_memory(value_states_transpose_arr, config.max_sqlen * config.embed_dim * sizeof(float));
    cache_num = new int[config.num_layers];
    for (int i = 0; i < config.num_layers; i++) cache_num[i] = 0;
    allocate_aligned_memory(query_states_unshape_arr, config.max_sqlen * config.embed_dim * sizeof(float));
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
}

inline void int4llamaAttention::shape(Matrix3D<float> unshape, Matrix3D<float> shaped, int sqlen) {
    PROFILE_START("int4llamaAttention::shape");
    assert(unshape.m_dim_x == 1);  // bsz == 1
    assert(unshape.m_dim_y == sqlen);
    assert(unshape.m_dim_z == this->num_heads * this->head_dim);
    assert(shaped.m_dim_x == this->num_heads);
    assert(shaped.m_dim_y == sqlen);
    assert(shaped.m_dim_z == this->head_dim);

    for (int i = 0; i < this->num_heads; i++) {
        for (int j = 0; j < sqlen; j++) {
            for (int k = 0; k < this->head_dim; k++) {
                shaped(i, j, k) = unshape(0, j, i * this->head_dim + k);
            }
        }
    }
    PROFILE_END("int4llamaAttention::shape");
}

inline void int4llamaAttention::unshape(Matrix3D<float> shaped, Matrix3D<float> unshape, int sqlen) {
    PROFILE_START("int4llamaAttention::unshpae");
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
    PROFILE_END("int4llamaAttention::unshpae");
}

int4llamaAttention::int4llamaAttention(std::string param_path, const struct model_config config) {
    int8_t *q_weight, *k_weight, *v_weight, *o_weight;
    allocate_aligned_memory(q_weight, (config.embed_dim * config.embed_dim * sizeof(int8_t)) / 2);
    allocate_aligned_memory(k_weight, (config.embed_dim * config.embed_dim * sizeof(int8_t)) / 2);
    allocate_aligned_memory(v_weight, (config.embed_dim * config.embed_dim * sizeof(int8_t)) / 2);
    allocate_aligned_memory(o_weight, (config.embed_dim * config.embed_dim * sizeof(int8_t)) / 2);
    this->q_proj = Linear_FP_int4(Matrix3D<int8_t>(q_weight, 1, config.embed_dim, config.embed_dim / 2),
                                  param_path + "/q_proj/weight.bin");
    this->k_proj = Linear_FP_int4(Matrix3D<int8_t>(k_weight, 1, config.embed_dim, config.embed_dim / 2),
                                  param_path + "/k_proj/weight.bin");
    this->v_proj = Linear_FP_int4(Matrix3D<int8_t>(v_weight, 1, config.embed_dim, config.embed_dim / 2),
                                  param_path + "/v_proj/weight.bin");
    this->o_proj = Linear_FP_int4(Matrix3D<int8_t>(o_weight, 1, config.embed_dim, config.embed_dim / 2),
                                  param_path + "/o_proj/weight.bin");

    float *cos_buf, *sin_buf;
    allocate_aligned_memory(cos_buf, config.max_sqlen * (config.embed_dim / config.num_heads) * sizeof(float));
    allocate_aligned_memory(sin_buf, config.max_sqlen * (config.embed_dim / config.num_heads) * sizeof(float));
    Matrix3D<float> cos(cos_buf, 1, config.max_sqlen, (config.embed_dim / config.num_heads));
    Matrix3D<float> sin(sin_buf, 1, config.max_sqlen, (config.embed_dim / config.num_heads));

    this->rotary_pos_emb = RotaryPosEmb(cos, sin, param_path + "/rotary_emb");

    float qk_bmm_alpha;
    read_to_array((param_path + "/qk_bmm/alpha.bin").c_str(), &qk_bmm_alpha, 1);
    this->qk_bmm = BMM_F32T(qk_bmm_alpha);
    this->pv_bmm = BMM_F32T(1.0f);

    this->embed_dim = config.embed_dim;
    this->num_heads = config.num_heads;
    assert(config.embed_dim % config.num_heads == 0);
    this->head_dim = config.embed_dim / config.num_heads;
}

struct int4llamaAttention_output int4llamaAttention::forward(const struct int4llamaAttention_input &input) {
    PROFILE_START(profile_name);
    struct int4llamaAttention_output output;
    const int sqlen = input.hidden_states.m_dim_y, b = input.hidden_states.m_dim_x;
    assert(b == 1);

    Matrix3D<float> query_states_unshape(query_states_unshape_arr, b, sqlen, embed_dim);
    // query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads,
    // self.head_dim).transpose(1, 2)
    this->q_proj.forward(input.hidden_states, query_states_unshape);
    Matrix3D<float> query_states(query_states_arr, this->num_heads, sqlen, this->head_dim);
    this->shape(query_states_unshape, query_states, sqlen);

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

    // key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_heads,
    // self.head_dim).transpose(1, 2)
    Matrix3D<float> key_states_unshape(key_states_unshape_arr, b, sqlen, embed_dim);
    this->k_proj.forward(input.hidden_states, key_states_unshape);
    Matrix3D<float> key_states(key_states_arr, this->num_heads, sqlen, this->head_dim);
    this->shape(key_states_unshape, key_states, sqlen);
    // value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_heads,
    // self.head_dim).transpose(1, 2)
    Matrix3D<float> value_states_unshape(value_states_unshape_arr, b, sqlen, embed_dim);
    this->v_proj.forward(input.hidden_states, value_states_unshape);
    Matrix3D<float> value_states(value_states_arr, this->num_heads, sqlen, this->head_dim);
    this->shape(value_states_unshape, value_states, sqlen);
    // print_first_k_elelment("query_states", query_states.m_data, 20);
    // print_first_k_elelment("key_states", key_states.m_data, 20);
    // print_first_k_elelment("value_states", value_states.m_data, 20);

    // printf("value_sum: %f, query_sum: %f, key_sum: %f\n", value_states.sum(),
    // query_states.sum(), key_states.sum()); apply_rotary_pos_emb
    int start_idx = 0;
    if (input.has_past_key_value) start_idx = input.past_key.m_dim_y;
    this->rotary_pos_emb.forward(query_states, key_states, start_idx, sqlen);
    // printf("(rotary)query_sum: %f, key_sum: %f\n", query_states.sum(),
    // key_states.sum());

    PROFILE_START(profile_name + "::cat_past_keys_values");
    int tgz = sqlen;
    if (input.has_past_key_value) {
        // # reuse k, v, self_attention
        // key_states = torch.cat([past_key_value[0], key_states], dim=2)
        // value_states = torch.cat([past_key_value[1], value_states], dim=2)
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
    // printf("(cat): past_key %f, past_value: %f\n", input.past_key.sum(),
    // input.past_value.sum()); printf("(cat): final_value_states_sum %f,
    // final_key_states: %f\n", final_value_states.sum(), final_key_states.sum());

    Matrix3D<float> attn_weights(attn_weights_arr, this->num_heads, sqlen, tgz);
    this->qk_bmm.forward(query_states, final_key_states, attn_weights);
    // printf("qk_bmm.forward, attn_weights.sum: %f\n", attn_weights.sum());
    // print_first_k_elelment("attn_weights", attn_weights.m_data, 20);

    // opt.py: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len,
    // src_len) + attention_mask
    batch_Add(attn_weights, input.attention_mask, attn_weights);
    // printf("batch_Add, attn_weights.sum: %f\n", attn_weights.sum());
    // Check for negative infinity, TODO: use multithread to speed up this
    for (int i = 0; i < attn_weights.length(); i++) {
        if (std::isinf(attn_weights.m_data[i])) {
            attn_weights.m_data[i] = std::numeric_limits<float>::lowest();
        }
    }
    // print_first_k_elelment("attn_weights(mask)", attn_weights.m_data, 20);

    Matrix3D<float> attn_probs(attn_weights_arr, this->num_heads, sqlen, tgz);
    softmax(attn_weights, attn_probs, 2);
    // printf("softmax, attn_probs.sum: %f\n", attn_probs.sum());
    // print_first_k_elelment("attn_probs", attn_probs.m_data, 20);

    Matrix3D<float> value_states_transpose(value_states_transpose_arr, this->num_heads, this->head_dim, tgz);
    transpose_1_2idx_float_threads(final_value_states, value_states_transpose, NUM_THREAD);

    Matrix3D<float> attn_output(attn_output_arr, this->num_heads, sqlen, this->head_dim);
    this->pv_bmm.forward(attn_probs, value_states_transpose, attn_output);
    // printf("pv_bmm.forward, attn_output.sum: %f\n", attn_output.sum());
    // print_first_k_elelment("attn_output", attn_output.m_data, 20);

    Matrix3D<float> attn_output_transpose(attn_output_transpose_arr, 1, sqlen, this->num_heads * this->head_dim);
    this->unshape(attn_output, attn_output_transpose, sqlen);

    Matrix3D<float> attn_output_fp(attn_output_fp_arr, 1, sqlen, this->num_heads * this->head_dim);
    this->o_proj.forward(attn_output_transpose, attn_output_fp);
    // printf("o_proj.forward, attn_output_fp.sum: %f\n", attn_output_fp.sum());
    // print_first_k_elelment("attn_output_fp", attn_output_fp.m_data, 20);
    // output assignment
    output.attn_output = attn_output_fp;
    output.past_key_value = {final_key_states, final_value_states};

    PROFILE_END(profile_name);
    return output;
}
