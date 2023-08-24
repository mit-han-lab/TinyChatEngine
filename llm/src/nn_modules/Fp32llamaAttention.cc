#include "Fp32llamaAttention.h"

#include <string.h>

#include <cmath>

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

struct transpose_1_2idx_float_arg {
    int start_idx, end_idx;
    Matrix3D<float> input, output;
};

void *transpose_1_2idx_float_func(void *args_) {
    struct transpose_1_2idx_float_arg *args = (struct transpose_1_2idx_float_arg *)args_;

    Matrix3D<float> input = args->input;
    Matrix3D<float> output = args->output;

    for (int i = 0; i < input.m_dim_x; i++) {
        for (int j = 0; j < input.m_dim_y; j++) {
            for (int k = args->start_idx; k < args->end_idx; k++) {
                output.m_data[i * output.m_dim_y * output.m_dim_z + k * output.m_dim_z + j] =
                    input.m_data[i * input.m_dim_y * input.m_dim_z + j * input.m_dim_z + k];
            }
        }
    }
    return NULL;
}

inline void transpose_1_2idx_float_threads(Matrix3D<float> &input, Matrix3D<float> &output) {
    PROFILE_START("Fp32llamaAttention::transpose_1_2idx_float");
    assert(input.m_dim_x == output.m_dim_x);
    assert(input.m_dim_y == output.m_dim_z);
    assert(input.m_dim_z == output.m_dim_y);

    if (input.m_dim_y == 1 || input.m_dim_z == 1) {
        memcpy(output.m_data, input.m_data, input.length() * sizeof(float));
    } else {
        int num_thread = NUM_THREAD;
        int loop_over_dim = input.m_dim_z;
        if (num_thread > loop_over_dim) num_thread = loop_over_dim;

        pthread_t thread_pool[NUM_THREAD];
        struct transpose_1_2idx_float_arg threads_args[NUM_THREAD];

        // Thread creation
        for (int j = 0; j < num_thread; j++) {
            threads_args[j].start_idx = j * (loop_over_dim / num_thread);
            threads_args[j].input = input;
            threads_args[j].output = output;
            if (j == num_thread - 1)
                threads_args[j].end_idx = loop_over_dim;
            else
                threads_args[j].end_idx = (j + 1) * (loop_over_dim / num_thread);
            pthread_create(&thread_pool[j], NULL, transpose_1_2idx_float_func, &threads_args[j]);
        }
        // Join threads
        for (int j = 0; j < num_thread; j++) {
            pthread_join(thread_pool[j], NULL);
        }
    }

    PROFILE_END("Fp32llamaAttention::transpose_1_2idx_float");
}

void Fp32llamaAttention::initialized_memory(const struct model_config config) {
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

inline void Fp32llamaAttention::shape(Matrix3D<float> unshape, Matrix3D<float> shaped, int sqlen) {
    PROFILE_START("Fp32llamaAttention::shape");
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
    PROFILE_END("Fp32llamaAttention::shape");
}

inline void Fp32llamaAttention::unshape(Matrix3D<float> shaped, Matrix3D<float> unshape, int sqlen) {
    PROFILE_START("Fp32llamaAttention::unshpae");
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
    PROFILE_END("Fp32llamaAttention::unshpae");
}

Fp32llamaAttention::Fp32llamaAttention(std::string param_path, const struct model_config config) {
    float *q_weight, *k_weight, *v_weight, *o_weight;
    allocate_aligned_memory(q_weight, config.embed_dim * config.embed_dim * sizeof(float));
    allocate_aligned_memory(k_weight, config.embed_dim * config.embed_dim * sizeof(float));
    allocate_aligned_memory(v_weight, config.embed_dim * config.embed_dim * sizeof(float));
    allocate_aligned_memory(o_weight, config.embed_dim * config.embed_dim * sizeof(float));
    this->q_proj =
        Linear_FP(Matrix3D<float>(q_weight, 1, config.embed_dim, config.embed_dim), param_path + "/q_proj/weight.bin");
    this->k_proj =
        Linear_FP(Matrix3D<float>(k_weight, 1, config.embed_dim, config.embed_dim), param_path + "/k_proj/weight.bin");
    this->v_proj =
        Linear_FP(Matrix3D<float>(v_weight, 1, config.embed_dim, config.embed_dim), param_path + "/v_proj/weight.bin");
    this->o_proj =
        Linear_FP(Matrix3D<float>(o_weight, 1, config.embed_dim, config.embed_dim), param_path + "/o_proj/weight.bin");

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

struct Fp32llamaAttention_output Fp32llamaAttention::forward(const struct Fp32llamaAttention_input &input) {
    PROFILE_START(profile_name);
    struct Fp32llamaAttention_output output;
    const int sqlen = input.hidden_states.m_dim_y, b = input.hidden_states.m_dim_x;
    assert(b == 1);

    // Query states
    Matrix3D<float> query_states_unshape(query_states_unshape_arr, b, sqlen, embed_dim);
    this->q_proj.forward(input.hidden_states, query_states_unshape);
    Matrix3D<float> query_states(query_states_arr, this->num_heads, sqlen, this->head_dim);
    this->shape(query_states_unshape, query_states, sqlen);

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

    // Key states
    Matrix3D<float> key_states_unshape(key_states_unshape_arr, b, sqlen, embed_dim);
    this->k_proj.forward(input.hidden_states, key_states_unshape);
    Matrix3D<float> key_states(key_states_arr, this->num_heads, sqlen, this->head_dim);
    this->shape(key_states_unshape, key_states, sqlen);

    // Value states
    Matrix3D<float> value_states_unshape(value_states_unshape_arr, b, sqlen, embed_dim);
    this->v_proj.forward(input.hidden_states, value_states_unshape);
    Matrix3D<float> value_states(value_states_arr, this->num_heads, sqlen, this->head_dim);
    this->shape(value_states_unshape, value_states, sqlen);

    // Rotate position
    int start_idx = 0;
    if (input.has_past_key_value) start_idx = input.past_key.m_dim_y;
    this->rotary_pos_emb.forward(query_states, key_states, start_idx, sqlen);

    // Concate with past key, value if exists
    PROFILE_START(profile_name + "::cat_past_keys_values");
    int tgz = sqlen;
    if (input.has_past_key_value) {
        // # reuse k, v, self_attention
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
    for (int i = 0; i < attn_weights.length(); i++) {
        if (std::isinf(attn_weights.m_data[i])) {
            attn_weights.m_data[i] = std::numeric_limits<float>::lowest();
        }
    }

    // Softmax QK
    Matrix3D<float> attn_probs(attn_weights_arr, this->num_heads, sqlen, tgz);
    softmax(attn_weights, attn_probs, 2);

    // Legacy implementation
    // Matrix3D<float> value_states_transpose(value_states_transpose_arr, this->num_heads, this->head_dim, tgz);
    // transpose_1_2idx_float_threads(final_value_states, value_states_transpose);
    // Matrix3D<float> attn_output(attn_output_arr, this->num_heads, sqlen, this->head_dim);
    // this->pv_bmm.forward(attn_probs, value_states_transpose, attn_output);

    // PV_BMM: This implementation avoid additional data movement and is much faster
    Matrix3D<float> attn_output(attn_output_arr, this->num_heads, sqlen, this->head_dim);
    this->pv_bmm.forward_weight_untransposed(attn_probs, final_value_states, attn_output);

    Matrix3D<float> attn_output_transpose(attn_output_transpose_arr, 1, sqlen, this->num_heads * this->head_dim);
    this->unshape(attn_output, attn_output_transpose, sqlen);

    // Output projection
    Matrix3D<float> attn_output_fp(attn_output_fp_arr, 1, sqlen, this->num_heads * this->head_dim);
    this->o_proj.forward(attn_output_transpose, attn_output_fp);

    // Output assignment
    output.attn_output = attn_output_fp;
    output.past_key_value = {final_key_states, final_value_states};

    PROFILE_END(profile_name);
    return output;
}
