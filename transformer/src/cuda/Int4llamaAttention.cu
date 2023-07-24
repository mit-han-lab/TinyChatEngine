#include <string.h>
#include <cmath>
#include <cfloat>

#include "Int4llamaAttention.h"
#include "operators.h"
#include "utils.h"
#include "utils.cuh"

static half *attn_weights_arr;
static half *attn_output_half_arr;
static half *query_states_unshape_arr;
static half *attn_output_arr;
static half *attn_output_transpose_arr;
static half *key_states_unshape_arr;
static half *key_states_arr;
static half *value_states_unshape_arr;
static half *value_states_arr;
static half *query_states_arr;
static half *value_states_transpose_arr;
static half *key_states_arr_cache;
static half *value_states_arr_cache;
static int *cache_num;

void Int4llamaAttention::initialized_memory(const struct model_config config) {
    allocate_aligned_memory_gpu(attn_weights_arr, config.num_heads * config.max_sqlen * config.max_sqlen * sizeof(half));
    allocate_aligned_memory_gpu(attn_output_half_arr, config.max_sqlen * config.embed_dim * sizeof(half));
    allocate_aligned_memory_gpu(attn_output_arr, config.max_sqlen * config.embed_dim * sizeof(half));
    allocate_aligned_memory_gpu(attn_output_transpose_arr, config.max_sqlen * config.embed_dim * sizeof(half));
    allocate_aligned_memory_gpu(key_states_unshape_arr, config.max_sqlen * config.embed_dim * sizeof(half));
    allocate_aligned_memory_gpu(key_states_arr, config.max_sqlen * config.embed_dim * sizeof(half));
    allocate_aligned_memory_gpu(value_states_unshape_arr, config.max_sqlen * config.embed_dim * sizeof(half));
    allocate_aligned_memory_gpu(value_states_arr, config.max_sqlen * config.embed_dim * sizeof(half));
    allocate_aligned_memory_gpu(query_states_arr, config.max_sqlen * config.embed_dim * sizeof(half));
    allocate_aligned_memory_gpu(value_states_transpose_arr, config.max_sqlen * config.embed_dim * sizeof(half));
    allocate_aligned_memory_gpu(query_states_unshape_arr, config.max_sqlen * config.embed_dim * sizeof(half));

    allocate_aligned_memory(cache_num, config.num_layers * sizeof(int));
    for (int i = 0; i < config.num_layers; i++) cache_num[i] = 0;

    allocate_aligned_memory_gpu(key_states_arr_cache, config.num_layers * 2 * config.max_sqlen * config.embed_dim * sizeof(half));
    allocate_aligned_memory_gpu(value_states_arr_cache, config.num_layers * 2 * config.max_sqlen * config.embed_dim * sizeof(half));
}

__global__ void shape_float(Matrix3D<float> unshape, Matrix3D<float> shaped, int num_heads, int sqlen, int head_dim) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    int k = threadIdx.z + blockIdx.z * blockDim.z;
    
    if (i < num_heads && j < sqlen && k < head_dim) {
        shaped(i, j, k) = unshape(0, j, i * head_dim + k);
    }
}

__global__ void shape_half(Matrix3D<half> unshape, Matrix3D<half> shaped, int num_heads, int sqlen, int head_dim) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    int k = threadIdx.z + blockIdx.z * blockDim.z;
    
    if (i < num_heads && j < sqlen && k < head_dim) {
        shaped(i, j, k) = unshape(0, j, i * head_dim + k);
    }
}

__global__ void unshape_float(Matrix3D<float> shaped, Matrix3D<float> unshape, int num_heads, int sqlen, int head_dim) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    int k = threadIdx.z + blockIdx.z * blockDim.z;
    
    if (i < num_heads && j < sqlen && k < head_dim) {
        unshape(0, j, i * head_dim + k) = shaped(i, j, k);
    }
}

__global__ void unshape_half(Matrix3D<half> shaped, Matrix3D<half> unshape, int num_heads, int sqlen, int head_dim) {
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
    this->q_proj = Linear_half_int4(Matrix3D<int>(q_weight, 1, config.embed_dim / 8, config.embed_dim),
                                  param_path + "/q_proj");
    this->k_proj = Linear_half_int4(Matrix3D<int>(k_weight, 1, config.embed_dim / 8, config.embed_dim),
                                  param_path + "/k_proj");
    this->v_proj = Linear_half_int4(Matrix3D<int>(v_weight, 1, config.embed_dim / 8, config.embed_dim),
                                  param_path + "/v_proj");
    this->o_proj = Linear_half_int4(Matrix3D<int>(o_weight, 1, config.embed_dim / 8, config.embed_dim),
                                  param_path + "/o_proj");

    float *cos_buf, *sin_buf;
    allocate_aligned_memory_gpu(cos_buf, config.max_sqlen * (config.embed_dim / config.num_heads) * sizeof(float));
    allocate_aligned_memory_gpu(sin_buf, config.max_sqlen * (config.embed_dim / config.num_heads) * sizeof(float));
    Matrix3D<float> cos(cos_buf, 1, config.max_sqlen, (config.embed_dim / config.num_heads));
    Matrix3D<float> sin(sin_buf, 1, config.max_sqlen, (config.embed_dim / config.num_heads));

    this->rotary_pos_emb = RotaryPosEmb_half(cos, sin, param_path + "/rotary_emb");

    float qk_bmm_alpha;
    read_to_array((param_path + "/qk_bmm/alpha.bin").c_str(), &qk_bmm_alpha, 1);
    this->qk_bmm = BMM_F16T(qk_bmm_alpha);
    this->pv_bmm = BMM_F16T(1.0f);

    this->embed_dim = config.embed_dim;
    this->num_heads = config.num_heads;
    assert(config.embed_dim % config.num_heads == 0);
    this->head_dim = config.embed_dim / config.num_heads;
    this->max_sqlen = config.max_sqlen;
}

__global__ void transpose_1_2idx_float(Matrix3D<float> input, Matrix3D<float> output) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i < input.m_dim_x && j < input.m_dim_y && k < input.m_dim_z) {
        output.m_data[i * output.m_dim_y * output.m_dim_z + k * output.m_dim_z + j] =
            input.m_data[i * input.m_dim_y * input.m_dim_z + j * input.m_dim_z + k];
    }
}

__global__ void transpose_1_2idx_half(Matrix3D<half> input, Matrix3D<half> output) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i < input.m_dim_x && j < input.m_dim_y && k < input.m_dim_z) {
        output.m_data[i * output.m_dim_y * output.m_dim_z + k * output.m_dim_z + j] =
            input.m_data[i * input.m_dim_y * input.m_dim_z + j * input.m_dim_z + k];
    }
}

__global__ void check_inf_float(Matrix3D<float> a) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (i < a.length()) {
        if (isinf(a.m_data[i])) {
            a.m_data[i] = -FLT_MAX;
        }
    }
}

__global__ void check_inf_half(Matrix3D<half> a) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (i < a.length()) {
        if (__hisinf(a.m_data[i]) == 1 || __hisinf(a.m_data[i]) == -1 || __hisnan(a.m_data[i])) {
            // a.m_data[i] = __float2half(-FLT_MAX);  // TODO: maybe could be optimized
            a.m_data[i] = __float2half(-65504.0f);  // TODO: maybe could be optimized
        }
    }
}

struct Int4llamaAttention_output Int4llamaAttention::forward(const struct Int4llamaAttention_input &input) {
    PROFILE_START(profile_name);
    struct Int4llamaAttention_output output;
    const int sqlen = input.hidden_states.m_dim_y, b = input.hidden_states.m_dim_x;
    assert(b == 1);

    // Query
    Matrix3D<half> query_states_unshape(query_states_unshape_arr, b, sqlen, embed_dim);
    this->q_proj.forward(input.hidden_states, query_states_unshape);

    Matrix3D<half> query_states(query_states_arr, this->num_heads, sqlen, this->head_dim);
    dim3 threadsPerBlock(8, 8, 8);
    dim3 numBlocks((this->num_heads + threadsPerBlock.x - 1) / threadsPerBlock.x,
                (sqlen + threadsPerBlock.y - 1) / threadsPerBlock.y,
                (this->head_dim + threadsPerBlock.z - 1) / threadsPerBlock.z);
    shape_half<<<numBlocks, threadsPerBlock>>>(query_states_unshape, query_states, this->num_heads, sqlen, this->head_dim);

    half *ret_value_states, *ret_key_states;
    if (cache_num[input.layer_idx] == 1) {
        ret_value_states = &value_states_arr_cache[(input.layer_idx * 2 + 1) * this->max_sqlen * this->embed_dim];
        ret_key_states = &key_states_arr_cache[(input.layer_idx * 2 + 1) * this->max_sqlen * this->embed_dim];
        cache_num[input.layer_idx] = 0;
    } else {
        ret_value_states = &value_states_arr_cache[input.layer_idx * 2 * this->max_sqlen * this->embed_dim];
        ret_key_states = &key_states_arr_cache[input.layer_idx * 2 * this->max_sqlen * this->embed_dim];
        cache_num[input.layer_idx] = 1;
    }

    // Key
    Matrix3D<half> key_states_unshape(key_states_unshape_arr, b, sqlen, embed_dim);
    this->k_proj.forward(input.hidden_states, key_states_unshape);

    Matrix3D<half> key_states(key_states_arr, this->num_heads, sqlen, this->head_dim);
    shape_half<<<numBlocks, threadsPerBlock>>>(key_states_unshape, key_states, this->num_heads, sqlen, this->head_dim);

    // Value
    Matrix3D<half> value_states_unshape(value_states_unshape_arr, b, sqlen, embed_dim);
    this->v_proj.forward(input.hidden_states, value_states_unshape);

    Matrix3D<half> value_states(value_states_arr, this->num_heads, sqlen, this->head_dim);
    shape_half<<<numBlocks, threadsPerBlock>>>(value_states_unshape, value_states, this->num_heads, sqlen, this->head_dim);

    int start_idx = 0;
    if (input.has_past_key_value) start_idx = input.past_key.m_dim_y;

    dim3 grid(num_heads, 1, 1);
    dim3 block(sqlen, 1, 1);
    RotaryPosEmb_half_forward<<<grid, block>>>(query_states, key_states, this->rotary_pos_emb.cos, this->rotary_pos_emb.sin, start_idx, sqlen);

    PROFILE_START(profile_name + "::cat_past_keys_values");
    int tgz = sqlen;
    if (input.has_past_key_value) {
        assert(input.past_key.m_dim_z == this->head_dim);
        tgz += input.past_key.m_dim_y;
        half *val_ptr = ret_value_states, *key_ptr = ret_key_states;
        int past_block = input.past_key.m_dim_y * input.past_key.m_dim_z;
        int sq_block = sqlen * this->head_dim;
        for (int i = 0; i < input.past_key.m_dim_x; i++) {
            cudaMemcpyAsync(val_ptr, &input.past_value.m_data[past_block * i], past_block * sizeof(half), cudaMemcpyHostToDevice);
            val_ptr += past_block;
            cudaMemcpyAsync(val_ptr, &value_states.m_data[sq_block * i], sq_block * sizeof(half), cudaMemcpyDeviceToDevice);
            val_ptr += sq_block;
            cudaMemcpyAsync(key_ptr, &input.past_key.m_data[past_block * i], past_block * sizeof(half), cudaMemcpyHostToDevice);
            key_ptr += past_block;
            cudaMemcpyAsync(key_ptr, &key_states.m_data[sq_block * i], sq_block * sizeof(half), cudaMemcpyDeviceToDevice);
            key_ptr += sq_block;
        }
    } else {
        cudaMemcpyAsync(ret_value_states, value_states_arr, (this->num_heads * tgz * this->head_dim) * sizeof(half), cudaMemcpyDeviceToDevice);
        cudaMemcpyAsync(ret_key_states, key_states_arr, (this->num_heads * tgz * this->head_dim) * sizeof(half), cudaMemcpyDeviceToDevice);
    }

    Matrix3D<half> final_value_states(ret_value_states, this->num_heads, tgz, this->head_dim);
    Matrix3D<half> final_key_states(ret_key_states, this->num_heads, tgz, this->head_dim);
    PROFILE_END(profile_name + "::cat_past_keys_values");

    Matrix3D<half> attn_weights(attn_weights_arr, this->num_heads, sqlen, tgz);
    this->qk_bmm.forward(query_states, final_key_states, attn_weights);

    dim3 numBlocks2((this->num_heads + threadsPerBlock.x - 1) / threadsPerBlock.x,
                (sqlen + threadsPerBlock.y - 1) / threadsPerBlock.y,
                (tgz + threadsPerBlock.z - 1) / threadsPerBlock.z);
    batch_Add_half<<<numBlocks2, threadsPerBlock>>>(attn_weights, input.attention_mask, attn_weights);

    int threadsPerBlock_1D = 1024;
    int blocksPerGrid_1D =(attn_weights.length() + threadsPerBlock_1D - 1) / threadsPerBlock_1D;
    check_inf_half<<<blocksPerGrid_1D, threadsPerBlock_1D>>>(attn_weights);

    Matrix3D<half> attn_probs(attn_weights_arr, this->num_heads, sqlen, tgz);
    int blockSize = 32;
    dim3 threadsPerBlock3(blockSize, blockSize);
    dim3 numBlocksPerGrid((this->num_heads + blockSize - 1) / blockSize, (sqlen + blockSize - 1) / blockSize);
    softmax_half<<<numBlocksPerGrid, threadsPerBlock3>>>(attn_weights, attn_probs);

    Matrix3D<half> value_states_transpose(value_states_transpose_arr, this->num_heads, this->head_dim, tgz);
    dim3 numBlocks4((this->num_heads + threadsPerBlock.x - 1) / threadsPerBlock.x,
                (tgz + threadsPerBlock.y - 1) / threadsPerBlock.y,
                (this->head_dim + threadsPerBlock.z - 1) / threadsPerBlock.z);
    transpose_1_2idx_half<<<numBlocks4, threadsPerBlock>>>(final_value_states, value_states_transpose);

    Matrix3D<half> attn_output(attn_output_arr, this->num_heads, sqlen, this->head_dim);
    this->pv_bmm.forward(attn_probs, value_states_transpose, attn_output);

    Matrix3D<half> attn_output_transpose(attn_output_transpose_arr, 1, sqlen, this->num_heads * this->head_dim);
    unshape_half<<<numBlocks, threadsPerBlock>>>(attn_output, attn_output_transpose, this->num_heads, sqlen, this->head_dim);

    Matrix3D<half> attn_output_half(attn_output_half_arr, 1, sqlen, this->num_heads * this->head_dim);
    this->o_proj.forward(attn_output_transpose, attn_output_half);

    // output assignment
    output.attn_output = attn_output_half;
    output.past_key_value = {final_key_states, final_value_states};

    PROFILE_END(profile_name);

    return output;
}