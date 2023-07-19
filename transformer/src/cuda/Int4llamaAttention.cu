#include "Int4llamaAttention.h"

#include <string.h>
#include <cmath>
#include <cfloat>

#include "operators.h"
#include "utils.h"

static float *attn_weights_arr;
static float *attn_output_fp_arr;
static float *query_states_unshape_arr;
static float *attn_output_arr;
static float *attn_output_transpose_arr;
static float *key_states_unshape_arr;
static float *key_states_arr;
static float *value_states_unshape_arr;
static float *value_states_arr;
static float *query_states_arr;
static float *value_states_transpose_arr;
//// Original code
// static float ***key_states_arr_cache;
// static float ***value_states_arr_cache;
//// CUDA 1
static float *key_states_arr_cache;
static float *value_states_arr_cache;
static int *cache_num;

//// CUDA code
// float *ret_value_states, *ret_key_states;

// __global__ void initializeCache(int *cache_num, int num_layers) {
//     int idx = threadIdx.x + blockIdx.x * blockDim.x;

//     if (idx < num_layers) {
//         cache_num[idx] = 0;
//     }
// }

__global__ void cacheSwap(int* cache_num, float* value_states_arr_cache, float* key_states_arr_cache, float* ret_value_states, float* ret_key_states, 
                          int layer_idx, int max_sqlen, int embed_dim) {
    int idx = threadIdx.x;
    if (idx != 0) return; // ensure only one thread does the assignment
    
    if (cache_num[layer_idx] == 1) {
        ret_value_states = &value_states_arr_cache[(layer_idx * 2 + 1) * max_sqlen * embed_dim];
        ret_key_states = &key_states_arr_cache[(layer_idx * 2 + 1) * max_sqlen * embed_dim];
        cache_num[layer_idx] = 0;
    } else {
        ret_value_states = &value_states_arr_cache[layer_idx * 2 * max_sqlen * embed_dim];
        ret_key_states = &key_states_arr_cache[layer_idx * 2 * max_sqlen * embed_dim];
        cache_num[layer_idx] = 1;
    }
}


void Int4llamaAttention::initialized_memory(const struct model_config config) {
    allocate_aligned_memory_gpu(attn_weights_arr, config.num_heads * config.max_sqlen * config.max_sqlen * sizeof(float));
    allocate_aligned_memory_gpu(attn_output_fp_arr, config.max_sqlen * config.embed_dim * sizeof(float));
    allocate_aligned_memory_gpu(attn_output_arr, config.max_sqlen * config.embed_dim * sizeof(float));
    allocate_aligned_memory_gpu(attn_output_transpose_arr, config.max_sqlen * config.embed_dim * sizeof(float));
    allocate_aligned_memory_gpu(key_states_unshape_arr, config.max_sqlen * config.embed_dim * sizeof(float));
    allocate_aligned_memory_gpu(key_states_arr, config.max_sqlen * config.embed_dim * sizeof(float));
    allocate_aligned_memory_gpu(value_states_unshape_arr, config.max_sqlen * config.embed_dim * sizeof(float));
    allocate_aligned_memory_gpu(value_states_arr, config.max_sqlen * config.embed_dim * sizeof(float));
    allocate_aligned_memory_gpu(query_states_arr, config.max_sqlen * config.embed_dim * sizeof(float));
    allocate_aligned_memory_gpu(value_states_transpose_arr, config.max_sqlen * config.embed_dim * sizeof(float));
    allocate_aligned_memory_gpu(query_states_unshape_arr, config.max_sqlen * config.embed_dim * sizeof(float));

    //// Original code
    // cache_num = new int[config.num_layers];
    // for (int i = 0; i < config.num_layers; i++) cache_num[i] = 0;
    // CUDA 1
    // allocate_aligned_memory_gpu(cache_num, config.num_layers * sizeof(int));
    allocate_aligned_memory(cache_num, config.num_layers * sizeof(int));
    for (int i = 0; i < config.num_layers; i++) cache_num[i] = 0;
    // int blockSize = 1024;
    // int numBlocks = (config.num_layers + blockSize - 1) / blockSize;
    // initializeCache<<<numBlocks, blockSize>>>(cache_num, config.num_layers);

    //// Original code
    // key_states_arr_cache = new float **[config.num_layers];
    // for (int i = 0; i < config.num_layers; ++i) {
    //     key_states_arr_cache[i] = new float *[2];
    //     for (int j = 0; j < 2; ++j) {
    //         allocate_aligned_memory_gpu(key_states_arr_cache[i][j], config.max_sqlen * config.embed_dim * sizeof(float));
    //     }
    // }
    // value_states_arr_cache = new float **[config.num_layers];
    // for (int i = 0; i < config.num_layers; ++i) {
    //     value_states_arr_cache[i] = new float *[2];
    //     for (int j = 0; j < 2; ++j) {
    //         allocate_aligned_memory_gpu(value_states_arr_cache[i][j], config.max_sqlen * config.embed_dim * sizeof(float));
    //     }
    // }
    //// CUDA 1
    allocate_aligned_memory_gpu(key_states_arr_cache, config.num_layers * 2 * config.max_sqlen * config.embed_dim * sizeof(float));
    allocate_aligned_memory_gpu(value_states_arr_cache, config.num_layers * 2 * config.max_sqlen * config.embed_dim * sizeof(float));
    // allocate_aligned_memory(key_states_arr_cache, config.num_layers * 2 * config.max_sqlen * config.embed_dim * sizeof(float));
    // allocate_aligned_memory(value_states_arr_cache, config.num_layers * 2 * config.max_sqlen * config.embed_dim * sizeof(float));

    //// CUDA code
    // allocate_aligned_memory_gpu(ret_value_states, config.max_sqlen * config.embed_dim * sizeof(float));
    // allocate_aligned_memory_gpu(ret_key_states, config.max_sqlen * config.embed_dim * sizeof(float));
}

inline void Int4llamaAttention::shape(Matrix3D<float> unshape, Matrix3D<float> shaped, int sqlen) {
    PROFILE_START("Int4llamaAttention::shape");
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
    PROFILE_END("Int4llamaAttention::shape");
}

__global__ void shape_half(Matrix3D<float> unshape, Matrix3D<float> shaped, int num_heads, int sqlen, int head_dim) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    int k = threadIdx.z + blockIdx.z * blockDim.z;
    
    if (i < num_heads && j < sqlen && k < head_dim) {
        shaped(i, j, k) = unshape(0, j, i * head_dim + k);
    }
}

inline void Int4llamaAttention::unshape(Matrix3D<float> shaped, Matrix3D<float> unshape, int sqlen) {
    PROFILE_START("Int4llamaAttention::unshape");
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
    PROFILE_END("Int4llamaAttention::unshape");
}

__global__ void unshape_half(Matrix3D<float> shaped, Matrix3D<float> unshape, int num_heads, int sqlen, int head_dim) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    int k = threadIdx.z + blockIdx.z * blockDim.z;
    
    if (i < num_heads && j < sqlen && k < head_dim) {
        unshape(0, j, i * head_dim + k) = shaped(i, j, k);
    }
}

Int4llamaAttention::Int4llamaAttention(std::string param_path, const struct model_config config) {
    allocate_aligned_memory_gpu(q_weight, (config.embed_dim * config.embed_dim * sizeof(int)) / 8);
    allocate_aligned_memory_gpu(k_weight, (config.embed_dim * config.embed_dim * sizeof(int)) / 8);
    allocate_aligned_memory_gpu(v_weight, (config.embed_dim * config.embed_dim * sizeof(int)) / 8);
    allocate_aligned_memory_gpu(o_weight, (config.embed_dim * config.embed_dim * sizeof(int)) / 8);
    this->q_proj = Linear_half_int4_ref(Matrix3D<int>(q_weight, 1, config.embed_dim / 8, config.embed_dim),
                                  param_path + "/q_proj");
    this->k_proj = Linear_half_int4_ref(Matrix3D<int>(k_weight, 1, config.embed_dim / 8, config.embed_dim),
                                  param_path + "/k_proj");
    this->v_proj = Linear_half_int4_ref(Matrix3D<int>(v_weight, 1, config.embed_dim / 8, config.embed_dim),
                                  param_path + "/v_proj");
    this->o_proj = Linear_half_int4_ref(Matrix3D<int>(o_weight, 1, config.embed_dim / 8, config.embed_dim),
                                  param_path + "/o_proj");

    float *cos_buf, *sin_buf;
    allocate_aligned_memory_gpu(cos_buf, config.max_sqlen * (config.embed_dim / config.num_heads) * sizeof(float));
    allocate_aligned_memory_gpu(sin_buf, config.max_sqlen * (config.embed_dim / config.num_heads) * sizeof(float));
    Matrix3D<float> cos(cos_buf, 1, config.max_sqlen, (config.embed_dim / config.num_heads));
    Matrix3D<float> sin(sin_buf, 1, config.max_sqlen, (config.embed_dim / config.num_heads));

    // this->rotary_pos_emb = RotaryPosEmb(cos, sin, param_path + "/rotary_emb");
    this->rotary_pos_emb = RotaryPosEmb_half(cos, sin, param_path + "/rotary_emb");

    float qk_bmm_alpha;
    read_to_array((param_path + "/qk_bmm/alpha.bin").c_str(), &qk_bmm_alpha, 1);
    this->qk_bmm = BMM_F16T(qk_bmm_alpha);
    // this->qk_bmm = BMM_F32T(qk_bmm_alpha);
    this->pv_bmm = BMM_F16T(1.0f);
    // this->pv_bmm = BMM_F32T(1.0f);

    this->embed_dim = config.embed_dim;
    this->num_heads = config.num_heads;
    assert(config.embed_dim % config.num_heads == 0);
    this->head_dim = config.embed_dim / config.num_heads;
    this->max_sqlen = config.max_sqlen;
}

struct transpose_1_2idx_float_arg {
    int start_idx, end_idx;
    Matrix3D<float> input, output;
};

static void *transpose_1_2idx_float_func(void *args_) {
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

static inline void transpose_1_2idx_float_threads(Matrix3D<float> &input, Matrix3D<float> &output) {
    PROFILE_START("Int8OPTAttention::transpose_1_2idx_float");
    assert(input.m_dim_x == output.m_dim_x);
    assert(input.m_dim_y == output.m_dim_z);
    assert(input.m_dim_z == output.m_dim_y);

    if (input.m_dim_y == 1 || input.m_dim_z == 1) {
        // memcpy(output.m_data, input.m_data, input.length() * sizeof(float));
        cudaMemcpy(output.m_data, input.m_data, input.length() * sizeof(float), cudaMemcpyDeviceToDevice);
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

    PROFILE_END("Int8OPTAttention::transpose_1_2idx_float");
}

__global__ void transpose_1_2idx_half(Matrix3D<float> input, Matrix3D<float> output) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i < input.m_dim_x && j < input.m_dim_y && k < input.m_dim_z) {
        output.m_data[i * output.m_dim_y * output.m_dim_z + k * output.m_dim_z + j] =
            input.m_data[i * input.m_dim_y * input.m_dim_z + j * input.m_dim_z + k];
    }
}

__global__ void check_inf_half(Matrix3D<float> a) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (i < a.length()) {
        if (isinf(a.m_data[i])) {
            a.m_data[i] = -FLT_MAX;
        }
    }
}

struct Int4llamaAttention_output Int4llamaAttention::forward(const struct Int4llamaAttention_input &input) {
    PROFILE_START(profile_name);
    struct Int4llamaAttention_output output;
    const int sqlen = input.hidden_states.m_dim_y, b = input.hidden_states.m_dim_x;
    assert(b == 1);

    Matrix3D<float> query_states_unshape(query_states_unshape_arr, b, sqlen, embed_dim);

    this->q_proj.forward(input.hidden_states, query_states_unshape);
    Matrix3D<float> query_states(query_states_arr, this->num_heads, sqlen, this->head_dim);

    //// Original code
    // this->shape(query_states_unshape, query_states, sqlen);
    //// CUDA 1
    dim3 threadsPerBlock(8, 8, 8);
    dim3 numBlocks((this->num_heads + threadsPerBlock.x - 1) / threadsPerBlock.x,
                (sqlen + threadsPerBlock.y - 1) / threadsPerBlock.y,
                (this->head_dim + threadsPerBlock.z - 1) / threadsPerBlock.z);
    shape_half<<<numBlocks, threadsPerBlock>>>(query_states_unshape, query_states, this->num_heads, sqlen, this->head_dim);
    // cudaDeviceSynchronize();

    // printf("00000000\n");
    //// Original code
    float *ret_value_states, *ret_key_states;
    // if (cache_num[input.layer_idx] == 1) {
    //     ret_value_states = value_states_arr_cache[input.layer_idx][1];
    //     ret_key_states = key_states_arr_cache[input.layer_idx][1];
    //     cache_num[input.layer_idx] = 0;
    // } else {
    //     ret_value_states = value_states_arr_cache[input.layer_idx][0];
    //     ret_key_states = key_states_arr_cache[input.layer_idx][0];
    //     cache_num[input.layer_idx] = 1;
    // }
    //// Modified 1
    if (cache_num[input.layer_idx] == 1) {
        ret_value_states = &value_states_arr_cache[(input.layer_idx * 2 + 1) * this->max_sqlen * this->embed_dim];
        ret_key_states = &key_states_arr_cache[(input.layer_idx * 2 + 1) * this->max_sqlen * this->embed_dim];
        cache_num[input.layer_idx] = 0;
    } else {
        ret_value_states = &value_states_arr_cache[input.layer_idx * 2 * this->max_sqlen * this->embed_dim];
        ret_key_states = &key_states_arr_cache[input.layer_idx * 2 * this->max_sqlen * this->embed_dim];
        cache_num[input.layer_idx] = 1;
    }
    //// CUDA 1
    // printf("11111111111\n"); 
    // cacheSwap<<<1, 1>>>(cache_num, value_states_arr_cache, key_states_arr_cache, ret_value_states, ret_key_states, input.layer_idx, this->max_sqlen, this->embed_dim);
    // cudaDeviceSynchronize();
    // printf("22222222222\n");

    Matrix3D<float> key_states_unshape(key_states_unshape_arr, b, sqlen, embed_dim);
    this->k_proj.forward(input.hidden_states, key_states_unshape);
    // printf("33333333333\n");
    Matrix3D<float> key_states(key_states_arr, this->num_heads, sqlen, this->head_dim);
    //// Original code
    // this->shape(key_states_unshape, key_states, sqlen);
    //// CUDA 1
    shape_half<<<numBlocks, threadsPerBlock>>>(key_states_unshape, key_states, this->num_heads, sqlen, this->head_dim);
    // cudaDeviceSynchronize();
    // printf("44444444444\n");

    Matrix3D<float> value_states_unshape(value_states_unshape_arr, b, sqlen, embed_dim);
    this->v_proj.forward(input.hidden_states, value_states_unshape);
    // printf("55555555555\n");
    Matrix3D<float> value_states(value_states_arr, this->num_heads, sqlen, this->head_dim);
    //// Original code
    // this->shape(value_states_unshape, value_states, sqlen);
    //// CUDA 1
    shape_half<<<numBlocks, threadsPerBlock>>>(value_states_unshape, value_states, this->num_heads, sqlen, this->head_dim);
    // cudaDeviceSynchronize();
    // printf("66666666666\n");

    int start_idx = 0;
    if (input.has_past_key_value) start_idx = input.past_key.m_dim_y;

    //// Original code
    // this->rotary_pos_emb.forward(query_states, key_states, start_idx, sqlen);
    //// CUDA 1
    // dim3 block(256);
    // dim3 grid((this->num_heads * this->rotary_pos_emb.cos.m_dim_z * sqlen + block.x - 1) / block.x);
    //// CUDA 2
    // int threads_per_block = min(sqlen, 1024);
    // int blocks_per_grid_x = num_heads;
    // int blocks_per_grid_y = (sqlen + threads_per_block - 1) / threads_per_block; // calculates ceil(len / threads_per_block)
    // dim3 grid(blocks_per_grid_x, blocks_per_grid_y, 1);
    // dim3 block(threads_per_block, 1, 1);
    //// CUDA 3
    dim3 grid(num_heads, 1, 1);
    dim3 block(sqlen, 1, 1);
    // printf("77777777777\n");
    RotaryPosEmb_half_forward<<<grid, block>>>(query_states, key_states, this->rotary_pos_emb.cos, this->rotary_pos_emb.sin, start_idx, sqlen);
    // cudaDeviceSynchronize();
    // printf("88888888888\n");


    PROFILE_START(profile_name + "::cat_past_keys_values");
    //// Original code
    int tgz = sqlen;
    // printf("input.past_key.m_dim_x: %d\n", input.past_key.m_dim_x);
    if (input.has_past_key_value) {
        assert(input.past_key.m_dim_z == this->head_dim);
        tgz += input.past_key.m_dim_y;
        float *val_ptr = ret_value_states, *key_ptr = ret_key_states;
        int past_block = input.past_key.m_dim_y * input.past_key.m_dim_z;
        int sq_block = sqlen * this->head_dim;
        for (int i = 0; i < input.past_key.m_dim_x; i++) {
            // memcpy(val_ptr, &input.past_value.m_data[past_block * i], past_block * sizeof(float));
            // val_ptr += past_block;
            // memcpy(val_ptr, &value_states.m_data[sq_block * i], sq_block * sizeof(float));
            // val_ptr += sq_block;
            // memcpy(key_ptr, &input.past_key.m_data[past_block * i], past_block * sizeof(float));
            // key_ptr += past_block;
            // memcpy(key_ptr, &key_states.m_data[sq_block * i], sq_block * sizeof(float));
            // key_ptr += sq_block;

            // cudaMemcpy(val_ptr, &input.past_value.m_data[past_block * i], past_block * sizeof(float), cudaMemcpyHostToDevice);
            // val_ptr += past_block;
            // cudaMemcpy(val_ptr, &value_states.m_data[sq_block * i], sq_block * sizeof(float), cudaMemcpyDeviceToDevice);
            // val_ptr += sq_block;
            // cudaMemcpy(key_ptr, &input.past_key.m_data[past_block * i], past_block * sizeof(float), cudaMemcpyHostToDevice);
            // key_ptr += past_block;
            // cudaMemcpy(key_ptr, &key_states.m_data[sq_block * i], sq_block * sizeof(float), cudaMemcpyDeviceToDevice);
            // key_ptr += sq_block;

            cudaMemcpyAsync(val_ptr, &input.past_value.m_data[past_block * i], past_block * sizeof(float), cudaMemcpyHostToDevice);
            val_ptr += past_block;
            cudaMemcpyAsync(val_ptr, &value_states.m_data[sq_block * i], sq_block * sizeof(float), cudaMemcpyDeviceToDevice);
            val_ptr += sq_block;
            cudaMemcpyAsync(key_ptr, &input.past_key.m_data[past_block * i], past_block * sizeof(float), cudaMemcpyHostToDevice);
            key_ptr += past_block;
            cudaMemcpyAsync(key_ptr, &key_states.m_data[sq_block * i], sq_block * sizeof(float), cudaMemcpyDeviceToDevice);
            key_ptr += sq_block;
        }
    } else {
        // printf("99999999999\n");
        // Put the data into the buffer
        // memcpy(ret_value_states, value_states_arr, (this->num_heads * tgz * this->head_dim) * sizeof(float));
        // memcpy(ret_key_states, key_states_arr, (this->num_heads * tgz * this->head_dim) * sizeof(float));

        // cudaMemcpy(ret_value_states, value_states_arr, (this->num_heads * tgz * this->head_dim) * sizeof(float), cudaMemcpyDeviceToDevice);
        // cudaMemcpy(ret_key_states, key_states_arr, (this->num_heads * tgz * this->head_dim) * sizeof(float), cudaMemcpyDeviceToDevice);

        cudaMemcpyAsync(ret_value_states, value_states_arr, (this->num_heads * tgz * this->head_dim) * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpyAsync(ret_key_states, key_states_arr, (this->num_heads * tgz * this->head_dim) * sizeof(float), cudaMemcpyDeviceToDevice);
        // cudaStreamSynchronize(0);

        // printf("aaaaaaaaaaa\n");
    }


    Matrix3D<float> final_value_states(ret_value_states, this->num_heads, tgz, this->head_dim);
    Matrix3D<float> final_key_states(ret_key_states, this->num_heads, tgz, this->head_dim);
    PROFILE_END(profile_name + "::cat_past_keys_values");

    Matrix3D<float> attn_weights(attn_weights_arr, this->num_heads, sqlen, tgz);

    // Only replace BMM_F32T with BMM_F16T, don't need to change the forward function here
    this->qk_bmm.forward(query_states, final_key_states, attn_weights);
    // printf("bbbbbbbbbbb\n");
    // dim3 block2(16, 16);  // You might want to optimize this size.
    // dim3 grid2((sqlen + block.x - 1) / block.x, (tgz + block.y - 1) / block.y);
    // BMM_F16T_forward<<<grid2, block2>>>(query_states, final_key_states, attn_weights);

    //// Original code
    // batch_Add(attn_weights, input.attention_mask, attn_weights);
    //// CUDA 1
    dim3 numBlocks2((this->num_heads + threadsPerBlock.x - 1) / threadsPerBlock.x,
                (sqlen + threadsPerBlock.y - 1) / threadsPerBlock.y,
                (tgz + threadsPerBlock.z - 1) / threadsPerBlock.z);
    batch_Add_half<<<numBlocks2, threadsPerBlock>>>(attn_weights, input.attention_mask, attn_weights);
    // cudaDeviceSynchronize();

    // Check for negative infinity, TODO: use multithread to speed up this
    //// Original code
    // for (int i = 0; i < attn_weights.length(); i++) {
    //     if (std::isinf(attn_weights.m_data[i])) {
    //         attn_weights.m_data[i] = std::numeric_limits<float>::lowest();
    //     }
    // }
    //// CUDA 1
    int threadsPerBlock_1D = 1024;
    int blocksPerGrid_1D =(attn_weights.length() + threadsPerBlock_1D - 1) / threadsPerBlock_1D;
    check_inf_half<<<blocksPerGrid_1D, threadsPerBlock_1D>>>(attn_weights);

    Matrix3D<float> attn_probs(attn_weights_arr, this->num_heads, sqlen, tgz);
    //// Original code
    // softmax(attn_weights, attn_probs, 2);
    //// CUDA 1
    // dim3 threadsPerBlock3(32, 32);
    // dim3 numBlocks3((this->num_heads + threadsPerBlock3.x - 1) / threadsPerBlock3.x,
    //             (sqlen + threadsPerBlock3.y - 1) / threadsPerBlock3.y);
    // softmax_half<<<numBlocks3, threadsPerBlock3>>>(attn_weights, attn_probs);
    //// CUDA 2
    int blockSize = 32;
    int numBlocks3 = (this->num_heads * sqlen + blockSize - 1) / blockSize;
    dim3 threadsPerBlock3(blockSize, blockSize);
    dim3 numBlocksPerGrid((this->num_heads + blockSize - 1) / blockSize, (sqlen + blockSize - 1) / blockSize);
    softmax_half<<<numBlocksPerGrid, threadsPerBlock3>>>(attn_weights, attn_probs);
    // cudaDeviceSynchronize();

    Matrix3D<float> value_states_transpose(value_states_transpose_arr, this->num_heads, this->head_dim, tgz);
    //// Original code
    // transpose_1_2idx_float_threads(final_value_states, value_states_transpose);
    //// CUDA 1
    dim3 numBlocks4((this->num_heads + threadsPerBlock.x - 1) / threadsPerBlock.x,
                (tgz + threadsPerBlock.y - 1) / threadsPerBlock.y,
                (this->head_dim + threadsPerBlock.z - 1) / threadsPerBlock.z);
    transpose_1_2idx_half<<<numBlocks4, threadsPerBlock>>>(final_value_states, value_states_transpose);
    // cudaDeviceSynchronize();

    Matrix3D<float> attn_output(attn_output_arr, this->num_heads, sqlen, this->head_dim);
    // Only replace BMM_F32T with BMM_F16T, don't need to change the forward function here
    this->pv_bmm.forward(attn_probs, value_states_transpose, attn_output);

    Matrix3D<float> attn_output_transpose(attn_output_transpose_arr, 1, sqlen, this->num_heads * this->head_dim);
    //// Original code
    // this->unshape(attn_output, attn_output_transpose, sqlen);
    //// CUDA 1
    unshape_half<<<numBlocks, threadsPerBlock>>>(attn_output, attn_output_transpose, this->num_heads, sqlen, this->head_dim);
    // cudaDeviceSynchronize();

    Matrix3D<float> attn_output_fp(attn_output_fp_arr, 1, sqlen, this->num_heads * this->head_dim);
    this->o_proj.forward(attn_output_transpose, attn_output_fp);

    // output assignment
    output.attn_output = attn_output_fp;
    output.past_key_value = {final_key_states, final_value_states};

    PROFILE_END(profile_name);

    // cudaFree(q_weight);
    // cudaFree(k_weight);
    // cudaFree(v_weight);
    // cudaFree(o_weight);

    return output;
}