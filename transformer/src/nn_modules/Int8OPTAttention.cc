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

    Matrix3D<int8_t> query_states_unshape(query_states_unshape_arr, b, sqlen, embed_dim);
    // opt.py: query_states = self.q_proj(hidden_states)
    this->q_proj.forward(input.hidden_states, query_states_unshape);
    // int8_t GT_query_states_unshape[query_states_unshape.length()];
    // print_first_k_elelment("input.hidden_states.m_data", input.hidden_states.m_data, 20);
    // print_first_k_elelment("q_proj.params.B.int8_data_ptr.m_data", q_proj.params.B.int8_data_ptr, 20);
    // print_first_k_elelment("query_states_unshape.m_data", query_states_unshape.m_data, 20);
    // printf("%.10f, %.10f\n", this->q_proj.alpha, this->q_proj.beta);
    // read_to_array("assets/tests/OPT_125m/Int8OPTAttention_value_states_query_states_unshape.bin",
    // GT_query_states_unshape, query_states_unshape.length()); assert(check_two_exact_equal(query_states_unshape_arr,
    // GT_query_states_unshape, query_states_unshape.length()));

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

    // opt.py: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    Matrix3D<int8_t> key_states_unshape(key_states_unshape_arr, b, sqlen, embed_dim);
    this->k_proj.forward(input.hidden_states, key_states_unshape);
    Matrix3D<int8_t> key_states(key_states_arr, this->num_heads, sqlen, this->head_dim);
    this->shpae(key_states_unshape, key_states, sqlen);

    // opt.py: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    Matrix3D<int8_t> value_states_unshape(value_states_unshape_arr, b, sqlen, embed_dim);
    this->v_proj.forward(input.hidden_states, value_states_unshape);
    Matrix3D<int8_t> value_states(value_states_arr, this->num_heads, sqlen, this->head_dim);
    this->shpae(value_states_unshape, value_states, sqlen);
    // print_first_k_elelment("key_states.m_data", key_states.m_data, 20);
    // print_first_k_elelment("value_states.m_data", value_states.m_data, 20);

    PROFILE_START(profile_name + "::cat_past_keys_values");
    int tgz = sqlen;
    if (input.has_past_key_value) {
        // # reuse k, v, self_attention
        // key_states = torch.cat([past_key_value[0], key_states], dim=2)
        // value_states = torch.cat([past_key_value[1], value_states], dim=2)
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

    // opt.py: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)

    Matrix3D<int8_t> query_states(query_states_arr, this->num_heads, sqlen, this->head_dim);
    this->shpae(query_states_unshape, query_states, sqlen);
    // print_first_k_elelment("query_states.m_data", query_states.m_data, 20);

    // opt.py: attn_weights = self.qk_bmm(query_states, key_states)
    // float attn_weights_arr[this->num_heads * sqlen * tgz];
    Matrix3D<float> attn_weights(attn_weights_arr, this->num_heads, sqlen, tgz);
    this->qk_bmm.forward(query_states, final_key_states, attn_weights);
    // print_first_k_elelment("attn_weights.m_data", attn_weights.m_data, 20);

    // opt.py: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    batch_Add(attn_weights, input.attention_mask, attn_weights);
    // print_first_k_elelment("attn_weights.m_data", attn_weights.m_data, 20);

    // opt.py: attn_probs = nn.functional.softmax(attn_weights, dim=-1)
    // float attn_probs_arr[this->num_heads * sqlen * sqlen];
    Matrix3D<float> attn_probs(attn_weights_arr, this->num_heads, sqlen, tgz);
    softmax(attn_weights, attn_probs, 2);
    // print_first_k_elelment("attn_probs.m_data", attn_probs.m_data, 20);

    // TODO: do we need layer_head_mask?

    // opt.py: attn_probs.mul_(127).round_()
    // opt.py: attn_probs = attn_probs.to(torch.int8)
    // int8_t attn_probs_int8_arr[this->num_heads * sqlen * tgz];
    PROFILE_START(profile_name + "::get_attn_probs_int8");
    Matrix3D<int8_t> attn_probs_int8(attn_probs_int8_arr, this->num_heads, sqlen, tgz);
    int len = attn_probs.length();
    for (int i = 0; i < len; i++) attn_probs_int8_arr[i] = static_cast<int8_t>(std::round(attn_probs.m_data[i] * 127));
    // print_first_k_elelment("attn_probs_int8.m_data", attn_probs_int8.m_data, 109);
    PROFILE_END(profile_name + "::get_attn_probs_int8");

    // opt.py: value_states = value_states.transpose(1, 2).contiguous()
    Matrix3D<int8_t> value_states_transpose(value_states_transpose_arr, this->num_heads, this->head_dim, tgz);
#ifdef USE_OPT_EXP
    transpose_1_2idx_threads(final_value_states, value_states_transpose);
#else
    transpose_1_2idx(final_value_states, value_states_transpose);
#endif
    // read_to_array("assets/tests/attn_probs_int8_mock.bin", attn_probs_int8.m_data, this->num_heads * sqlen * tgz);
    // read_to_array("assets/tests/value_states_transpose_mock.bin", value_states_transpose.m_data, this->num_heads *
    // tgz * this->head_dim); opt.py: attn_output = self.pv_bmm(attn_probs, value_states)
    Matrix3D<int8_t> attn_output(attn_output_arr, this->num_heads, sqlen, this->head_dim);
    this->pv_bmm.forward(attn_probs_int8, value_states_transpose, attn_output);
    // print_first_k_elelment("attn_output", attn_output.m_data, 20);

    // opt.py: attn_output = attn_output.transpose(1, 2)
    // opt.py: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim).contiguous()
    Matrix3D<int8_t> attn_output_transpose(attn_output_transpose_arr, 1, sqlen, this->num_heads * this->head_dim);
    this->unshape(attn_output, attn_output_transpose, sqlen);

    // read_to_array("assets/tests/OPT_1.3B/layer23_attn_output_before_outproj.bin", attn_output_transpose.m_data,
    // attn_output_transpose.length()); print_first_k_elelment("attn_output_transpose.m_data",
    // attn_output_transpose.m_data, 20);
    Matrix3D<float> attn_output_fp(attn_output_fp_arr, 1, sqlen, this->num_heads * this->head_dim);
    this->out_proj.forward(attn_output_transpose, attn_output_fp);
    // output assignment
    output.attn_output = attn_output_fp;
    output.past_key_value = {final_key_states, final_value_states};

    PROFILE_END(profile_name);
    return output;
}
