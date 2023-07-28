#include "Int8OPTAttention.h"

#include <string.h>

#include <cmath>

#include "operators.h"
#include "utils.h"

static float *attn_weights_arr;
static float *attn_probs_arr;
static int8_t *attn_probs_int8_arr;
static int8_t ***key_states_arr_cache;
static int8_t ***value_states_arr_cache;
static float *attn_output_fp_arr;
static int *cache_num;
static int8_t *query_states_unshape_arr;
static int8_t *attn_output_arr;
static int8_t *attn_output_transpose_arr;
static int8_t *key_states_unshape_arr;
static int8_t *key_states_arr;
static int8_t *value_states_unshape_arr;
static int8_t *value_states_arr;
static int8_t *query_states_arr;
static int8_t *value_states_transpose_arr;

void Int8OPTAttention::initialized_memory(const struct model_config config) {
    allocate_aligned_memory(attn_weights_arr, config.num_heads * config.max_sqlen * config.max_sqlen * sizeof(float));
    allocate_aligned_memory(attn_probs_arr, config.num_heads * config.max_sqlen * config.max_sqlen * sizeof(float));
    allocate_aligned_memory(attn_probs_int8_arr,
                            config.num_heads * config.max_sqlen * config.max_sqlen * sizeof(int8_t));
    allocate_aligned_memory(attn_output_fp_arr, config.max_sqlen * config.embed_dim * sizeof(float));
    allocate_aligned_memory(attn_output_arr, config.max_sqlen * config.embed_dim * sizeof(int8_t));
    allocate_aligned_memory(attn_output_transpose_arr, config.max_sqlen * config.embed_dim * sizeof(int8_t));
    allocate_aligned_memory(key_states_unshape_arr, config.max_sqlen * config.embed_dim * sizeof(int8_t));
    allocate_aligned_memory(key_states_arr, config.max_sqlen * config.embed_dim * sizeof(int8_t));
    allocate_aligned_memory(value_states_unshape_arr, config.max_sqlen * config.embed_dim * sizeof(int8_t));
    allocate_aligned_memory(value_states_arr, config.max_sqlen * config.embed_dim * sizeof(int8_t));
    allocate_aligned_memory(query_states_arr, config.max_sqlen * config.embed_dim * sizeof(int8_t));
    allocate_aligned_memory(value_states_transpose_arr, config.max_sqlen * config.embed_dim * sizeof(int8_t));
    cache_num = new int[config.num_layers];
    for (int i = 0; i < config.num_layers; i++) cache_num[i] = 0;
    allocate_aligned_memory(query_states_unshape_arr, config.max_sqlen * config.embed_dim * sizeof(int8_t));
    key_states_arr_cache = new int8_t **[config.num_layers];
    for (int i = 0; i < config.num_layers; ++i) {
        key_states_arr_cache[i] = new int8_t *[2];
        for (int j = 0; j < 2; ++j) {
            allocate_aligned_memory(key_states_arr_cache[i][j], config.max_sqlen * config.embed_dim * sizeof(int8_t));
        }
    }
    value_states_arr_cache = new int8_t **[config.num_layers];
    for (int i = 0; i < config.num_layers; ++i) {
        value_states_arr_cache[i] = new int8_t *[2];
        for (int j = 0; j < 2; ++j) {
            allocate_aligned_memory(value_states_arr_cache[i][j], config.max_sqlen * config.embed_dim * sizeof(int8_t));
        }
    }
}

Int8OPTAttention::Int8OPTAttention(std::string param_path, const struct model_config config, BMM_S8T_S8N_F32T &qk_bmm,
                                   BMM_S8T_S8N_S8T &pv_bmm, W8A8B8O8Linear &k_proj, W8A8B8O8Linear &v_proj,
                                   W8A8B8O8Linear &q_proj, W8A8BFP32OFP32Linear &out_proj) {
    load_BMM_S8T_S8N_F32T(qk_bmm, param_path + "/qk_bmm");
    load_BMM_S8T_S8N_S8T(pv_bmm, param_path + "/pv_bmm");
    load_W8A8B8O8Linear_params(k_proj, param_path + "/k_proj");
    load_W8A8B8O8Linear_params(v_proj, param_path + "/v_proj");
    load_W8A8B8O8Linear_params(q_proj, param_path + "/q_proj");
    load_W8A8BFP32OFP32Linear_params(out_proj, param_path + "/out_proj");

    this->embed_dim = config.embed_dim;
    this->num_heads = config.num_heads;
    assert(config.embed_dim % config.num_heads == 0);
    this->head_dim = config.embed_dim / config.num_heads;
    this->qk_bmm = qk_bmm;
    this->pv_bmm = pv_bmm;
    this->k_proj = k_proj;
    this->v_proj = v_proj;
    this->q_proj = q_proj;
    this->out_proj = out_proj;
}

inline void Int8OPTAttention::shpae(Matrix3D<int8_t> unshape, Matrix3D<int8_t> shaped, int sqlen) {
    PROFILE_START("Int8OPTAttention::shpae");
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
    PROFILE_END("Int8OPTAttention::shpae");
}

inline void Int8OPTAttention::unshape(Matrix3D<int8_t> shaped, Matrix3D<int8_t> unshape, int sqlen) {
    PROFILE_START("Int8OPTAttention::unshpae");
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
    PROFILE_END("Int8OPTAttention::unshpae");
}

struct transpose_1_2idx_arg {
    int start_idx, end_idx;
    Matrix3D<int8_t> input, output;
};

void *transpose_1_2idx_func(void *args_) {
    struct transpose_1_2idx_arg *args = (struct transpose_1_2idx_arg *)args_;

    Matrix3D<int8_t> input = args->input;
    Matrix3D<int8_t> output = args->output;

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

inline void transpose_1_2idx_threads(Matrix3D<int8_t> &input, Matrix3D<int8_t> &output) {
    PROFILE_START("Int8OPTAttention::transpose_1_2idx");
    assert(input.m_dim_x == output.m_dim_x);
    assert(input.m_dim_y == output.m_dim_z);
    assert(input.m_dim_z == output.m_dim_y);

    if (input.m_dim_y == 1 || input.m_dim_z == 1) {
        memcpy(output.m_data, input.m_data, input.length() * sizeof(int8_t));
    } else {
        int num_thread = NUM_THREAD;
        int loop_over_dim = input.m_dim_z;
        if (num_thread > loop_over_dim) num_thread = loop_over_dim;

        pthread_t thread_pool[NUM_THREAD];
        struct transpose_1_2idx_arg threads_args[NUM_THREAD];

        // Thread creation
        for (int j = 0; j < num_thread; j++) {
            threads_args[j].start_idx = j * (loop_over_dim / num_thread);
            threads_args[j].input = input;
            threads_args[j].output = output;
            if (j == num_thread - 1)
                threads_args[j].end_idx = loop_over_dim;
            else
                threads_args[j].end_idx = (j + 1) * (loop_over_dim / num_thread);
            pthread_create(&thread_pool[j], NULL, transpose_1_2idx_func, &threads_args[j]);
        }
        // Join threads
        for (int j = 0; j < num_thread; j++) {
            pthread_join(thread_pool[j], NULL);
        }
    }

    PROFILE_END("Int8OPTAttention::transpose_1_2idx");
}

template <typename T>
inline void transpose_1_2idx(Matrix3D<T> &input, Matrix3D<T> &output) {
    PROFILE_START("Int8OPTAttention::transpose_1_2idx");
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
    PROFILE_END("Int8OPTAttention::transpose_1_2idx");
}

struct Int8OPTAttention_output Int8OPTAttention::forward(const struct Int8OPTAttention_input &input) {
    PROFILE_START(profile_name);
    struct Int8OPTAttention_output output;
    const int sqlen = input.hidden_states.m_dim_y, b = input.hidden_states.m_dim_x;
    assert(b == 1);

    // Query states
    Matrix3D<int8_t> query_states_unshape(query_states_unshape_arr, b, sqlen, embed_dim);
    this->q_proj.forward(input.hidden_states, query_states_unshape);

    // Get the memory buffer
    int8_t *ret_value_states, *ret_key_states;
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
    Matrix3D<int8_t> key_states_unshape(key_states_unshape_arr, b, sqlen, embed_dim);
    this->k_proj.forward(input.hidden_states, key_states_unshape);
    Matrix3D<int8_t> key_states(key_states_arr, this->num_heads, sqlen, this->head_dim);
    this->shpae(key_states_unshape, key_states, sqlen);

    // Value states
    Matrix3D<int8_t> value_states_unshape(value_states_unshape_arr, b, sqlen, embed_dim);
    this->v_proj.forward(input.hidden_states, value_states_unshape);
    Matrix3D<int8_t> value_states(value_states_arr, this->num_heads, sqlen, this->head_dim);
    this->shpae(value_states_unshape, value_states, sqlen);

    // Concate with past key, value if exists
    PROFILE_START(profile_name + "::cat_past_keys_values");
    int tgz = sqlen;
    if (input.has_past_key_value) {
        assert(input.past_key.m_dim_z == this->head_dim);
        tgz += input.past_key.m_dim_y;
        int8_t *val_ptr = ret_value_states, *key_ptr = ret_key_states;
        int past_block = input.past_key.m_dim_y * input.past_key.m_dim_z;
        int sq_block = sqlen * this->head_dim;
        for (int i = 0; i < input.past_key.m_dim_x; i++) {
            memcpy(val_ptr, &input.past_value.m_data[past_block * i], past_block * sizeof(int8_t));
            val_ptr += past_block;
            memcpy(val_ptr, &value_states.m_data[sq_block * i], sq_block * sizeof(int8_t));
            val_ptr += sq_block;
            memcpy(key_ptr, &input.past_key.m_data[past_block * i], past_block * sizeof(int8_t));
            key_ptr += past_block;
            memcpy(key_ptr, &key_states.m_data[sq_block * i], sq_block * sizeof(int8_t));
            key_ptr += sq_block;
        }
    } else {
        // Put the data into the buffer
        memcpy(ret_value_states, value_states_arr, (this->num_heads * tgz * this->head_dim) * sizeof(int8_t));
        memcpy(ret_key_states, key_states_arr, (this->num_heads * tgz * this->head_dim) * sizeof(int8_t));
    }
    Matrix3D<int8_t> final_value_states(ret_value_states, this->num_heads, tgz, this->head_dim);
    Matrix3D<int8_t> final_key_states(ret_key_states, this->num_heads, tgz, this->head_dim);
    PROFILE_END(profile_name + "::cat_past_keys_values");

    Matrix3D<int8_t> query_states(query_states_arr, this->num_heads, sqlen, this->head_dim);
    this->shpae(query_states_unshape, query_states, sqlen);

    // QK_BMM
    Matrix3D<float> attn_weights(attn_weights_arr, this->num_heads, sqlen, tgz);
    this->qk_bmm.forward(query_states, final_key_states, attn_weights);

    // Add mask
    batch_Add(attn_weights, input.attention_mask, attn_weights);

    // Softmax QK
    Matrix3D<float> attn_probs(attn_weights_arr, this->num_heads, sqlen, tgz);
    softmax(attn_weights, attn_probs, 2);

    PROFILE_START(profile_name + "::get_attn_probs_int8");
    Matrix3D<int8_t> attn_probs_int8(attn_probs_int8_arr, this->num_heads, sqlen, tgz);
    int len = attn_probs.length();
    for (int i = 0; i < len; i++) attn_probs_int8_arr[i] = static_cast<int8_t>(std::round(attn_probs.m_data[i] * 127));
    PROFILE_END(profile_name + "::get_attn_probs_int8");

    // Transpose V for PV_BMM
    Matrix3D<int8_t> value_states_transpose(value_states_transpose_arr, this->num_heads, this->head_dim, tgz);
    transpose_1_2idx(final_value_states, value_states_transpose);
    Matrix3D<int8_t> attn_output(attn_output_arr, this->num_heads, sqlen, this->head_dim);
    this->pv_bmm.forward(attn_probs_int8, value_states_transpose, attn_output);

    Matrix3D<int8_t> attn_output_transpose(attn_output_transpose_arr, 1, sqlen, this->num_heads * this->head_dim);
    this->unshape(attn_output, attn_output_transpose, sqlen);

    // Output projection
    Matrix3D<float> attn_output_fp(attn_output_fp_arr, 1, sqlen, this->num_heads * this->head_dim);
    this->out_proj.forward(attn_output_transpose, attn_output_fp);

    // Output assignment
    output.attn_output = attn_output_fp;
    output.past_key_value = {final_key_states, final_value_states};

    PROFILE_END(profile_name);
    return output;
}
