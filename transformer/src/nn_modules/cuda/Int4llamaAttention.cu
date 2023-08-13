#include <string.h>
#include <cmath>
#include <cfloat>

#include "Int4llamaAttention.h"
#include "operators.h"
#include "utils.h"

static float16_t *attn_weights_arr = nullptr;
static float16_t *attn_output_half_arr = nullptr;
static float16_t *query_states_unshape_arr = nullptr;
static float16_t *attn_output_arr = nullptr;
static float16_t *attn_output_transpose_arr = nullptr;
static float16_t *key_states_unshape_arr = nullptr;
static float16_t *key_states_arr = nullptr;
static float16_t *value_states_unshape_arr = nullptr;
static float16_t *value_states_arr = nullptr;
static float16_t *query_states_arr = nullptr;
static float16_t *value_states_transpose_arr = nullptr;
static float16_t *key_states_arr_cache = nullptr;
static float16_t *value_states_arr_cache = nullptr;
static int *cache_num = nullptr;

// static float16_t* split_8_buffer;

void Int4llamaAttention::initialized_memory(const struct model_config config) {
    allocate_aligned_memory_gpu(attn_weights_arr, config.num_heads * config.max_sqlen * config.max_sqlen * sizeof(float16_t));
    allocate_aligned_memory_gpu(attn_output_half_arr, config.max_sqlen * config.embed_dim * sizeof(float16_t));
    allocate_aligned_memory_gpu(attn_output_arr, config.max_sqlen * config.embed_dim * sizeof(float16_t));
    allocate_aligned_memory_gpu(attn_output_transpose_arr, config.max_sqlen * config.embed_dim * sizeof(float16_t));
    allocate_aligned_memory_gpu(key_states_unshape_arr, config.max_sqlen * config.embed_dim * sizeof(float16_t));
    allocate_aligned_memory_gpu(key_states_arr, config.max_sqlen * config.embed_dim * sizeof(float16_t));
    allocate_aligned_memory_gpu(value_states_unshape_arr, config.max_sqlen * config.embed_dim * sizeof(float16_t));
    allocate_aligned_memory_gpu(value_states_arr, config.max_sqlen * config.embed_dim * sizeof(float16_t));
    allocate_aligned_memory_gpu(query_states_arr, config.max_sqlen * config.embed_dim * sizeof(float16_t));
    allocate_aligned_memory_gpu(value_states_transpose_arr, config.max_sqlen * config.embed_dim * sizeof(float16_t));
    allocate_aligned_memory_gpu(query_states_unshape_arr, config.max_sqlen * config.embed_dim * sizeof(float16_t));

    allocate_aligned_memory(cache_num, config.num_layers * sizeof(int));
    for (int i = 0; i < config.num_layers; i++) cache_num[i] = 0;

    allocate_aligned_memory_gpu(key_states_arr_cache, config.num_layers * 2 * config.max_sqlen * config.embed_dim * sizeof(float16_t));
    allocate_aligned_memory_gpu(value_states_arr_cache, config.num_layers * 2 * config.max_sqlen * config.embed_dim * sizeof(float16_t));

    // allocate_aligned_memory_gpu(split_8_buffer, config.max_sqlen * config.embed_dim * sizeof(float16_t) * 8);
    allocate_aligned_memory_gpu(split_8_buffer, config.max_sqlen * config.vocsize * sizeof(float16_t) * 8);
    // printf("config.max_sqlen, config.embed_dim: %d, %d\n", config.max_sqlen, config.embed_dim);
}

template <typename T>
__global__ void shape_cuda(Matrix3D<T> unshape, Matrix3D<T> shaped, int num_heads, int sqlen, int head_dim) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    int k = threadIdx.z + blockIdx.z * blockDim.z;
    
    if (i < num_heads && j < sqlen && k < head_dim) {
        shaped(i, j, k) = unshape(0, j, i * head_dim + k);
    }
}

template <typename T>
__global__ void unshape_cuda(Matrix3D<T> shaped, Matrix3D<T> unshape, int num_heads, int sqlen, int head_dim) {
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

    allocate_aligned_memory_gpu(cos_buf, config.max_sqlen * (config.embed_dim / config.num_heads) * sizeof(half));
    allocate_aligned_memory_gpu(sin_buf, config.max_sqlen * (config.embed_dim / config.num_heads) * sizeof(half));
    Matrix3D<half> cos(cos_buf, 1, config.max_sqlen, (config.embed_dim / config.num_heads));
    Matrix3D<half> sin(sin_buf, 1, config.max_sqlen, (config.embed_dim / config.num_heads));

    this->rotary_pos_emb = RotaryPosEmb_cuda(cos, sin, param_path + "/rotary_emb");

    half qk_bmm_alpha;
    read_to_array_half((param_path + "/qk_bmm/alpha_half.bin").c_str(), &qk_bmm_alpha, 1);
    this->qk_bmm = BMM_F16T(qk_bmm_alpha);
    this->pv_bmm = BMM_F16T(__float2half(1.0f));

    this->embed_dim = config.embed_dim;
    this->num_heads = config.num_heads;
    assert(config.embed_dim % config.num_heads == 0);
    this->head_dim = config.embed_dim / config.num_heads;
    this->max_sqlen = config.max_sqlen;
}

template <typename T>
__global__ void transpose_1_2idx_cuda(Matrix3D<T> input, Matrix3D<T> output) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i < input.m_dim_x && j < input.m_dim_y && k < input.m_dim_z) {
        output.m_data[i * output.m_dim_y * output.m_dim_z + k * output.m_dim_z + j] =
            input.m_data[i * input.m_dim_y * input.m_dim_z + j * input.m_dim_z + k];
    }
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

__global__ void transpose_1_2idx_half(Matrix3D<float16_t> input, Matrix3D<float16_t> output) {
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

__global__ void check_inf_half(Matrix3D<float16_t> a) {
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

    // printf("aaaaaaaaaa\n");

    // cudaEvent_t start_Attention, stop_Attention, start, stop;
    // cudaEventCreate(&start_Attention);
    // cudaEventCreate(&stop_Attention);
    // cudaEventCreate(&start);
    // cudaEventCreate(&stop);
    // float milliseconds = 0;

    // cudaEventRecord(start_Attention);

    struct Int4llamaAttention_output output;
    const int sqlen = input.hidden_states.m_dim_y, b = input.hidden_states.m_dim_x;
    assert(b == 1);

    // Query
    // cudaEventRecord(start);
    // PROFILE_START(profile_name + "::q_proj.forward");
    Matrix3D<float16_t> query_states_unshape(query_states_unshape_arr, b, sqlen, embed_dim);
    this->q_proj.forward(input.hidden_states, query_states_unshape, split_8_buffer);
    // PROFILE_END(profile_name + "::q_proj.forward");
    // cudaEventRecord(stop);
    // cudaEventSynchronize(stop);
    // cudaEventElapsedTime(&milliseconds, start, stop);
    // printf("q_proj.forward: %.2f ms\n", milliseconds * this->num_heads);

    // cudaEventRecord(start);
    // PROFILE_START(profile_name + "::shape_cuda_query");
    Matrix3D<float16_t> query_states(query_states_arr, this->num_heads, sqlen, this->head_dim);
    dim3 threadsPerBlock(8, 4, 32);
    dim3 numBlocks((this->num_heads + threadsPerBlock.x - 1) / threadsPerBlock.x,
                (sqlen + threadsPerBlock.y - 1) / threadsPerBlock.y,
                (this->head_dim + threadsPerBlock.z - 1) / threadsPerBlock.z);
    shape_cuda<<<numBlocks, threadsPerBlock>>>(query_states_unshape, query_states, this->num_heads, sqlen, this->head_dim);
    // PROFILE_END(profile_name + "::shape_cuda_query");
    // cudaEventRecord(stop);
    // cudaEventSynchronize(stop);
    // cudaEventElapsedTime(&milliseconds, start, stop);
    // printf("shape_cuda of q_proj: %.2f ms\n", milliseconds * this->num_heads);

    // cudaEventRecord(start);
    // PROFILE_START(profile_name + "::set_cache_num");
    // printf("bbbbbbbbbb\n");
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
    // PROFILE_END(profile_name + "::set_cache_num");
    // cudaEventRecord(stop);
    // cudaEventSynchronize(stop);
    // cudaEventElapsedTime(&milliseconds, start, stop);
    // printf("set_cache_num: %.2f ms\n", milliseconds * this->num_heads);
    // printf("cccccccccc\n");

    // Key
    // cudaEventRecord(start);
    // PROFILE_START(profile_name + "::k_proj.forward");
    Matrix3D<float16_t> key_states_unshape(key_states_unshape_arr, b, sqlen, embed_dim);
    this->k_proj.forward(input.hidden_states, key_states_unshape, split_8_buffer);
    // PROFILE_END(profile_name + "::k_proj.forward");
    // cudaEventRecord(stop);
    // cudaEventSynchronize(stop);
    // cudaEventElapsedTime(&milliseconds, start, stop);
    // printf("k_proj.forward: %.2f ms\n", milliseconds * this->num_heads);

    // cudaEventRecord(start);
    // PROFILE_START(profile_name + "::shape_cuda_key");
    Matrix3D<float16_t> key_states(key_states_arr, this->num_heads, sqlen, this->head_dim);
    shape_cuda<<<numBlocks, threadsPerBlock>>>(key_states_unshape, key_states, this->num_heads, sqlen, this->head_dim);
    // PROFILE_END(profile_name + "::shape_cuda_key");
    // cudaEventRecord(stop);
    // cudaEventSynchronize(stop);
    // cudaEventElapsedTime(&milliseconds, start, stop);
    // printf("shape_cuda of k_proj: %.2f ms\n", milliseconds * this->num_heads);

    // Value
    // cudaEventRecord(start);
    // PROFILE_START(profile_name + "::v_proj.forward");
    Matrix3D<float16_t> value_states_unshape(value_states_unshape_arr, b, sqlen, embed_dim);
    this->v_proj.forward(input.hidden_states, value_states_unshape, split_8_buffer);
    // PROFILE_END(profile_name + "::v_proj.forward");
    // cudaEventRecord(stop);
    // cudaEventSynchronize(stop);
    // cudaEventElapsedTime(&milliseconds, start, stop);
    // printf("v_proj.forward: %.2f ms\n", milliseconds * this->num_heads);

    // cudaEventRecord(start);
    // PROFILE_START(profile_name + "::shape_cuda_value");
    Matrix3D<float16_t> value_states(value_states_arr, this->num_heads, sqlen, this->head_dim);
    shape_cuda<<<numBlocks, threadsPerBlock>>>(value_states_unshape, value_states, this->num_heads, sqlen, this->head_dim);
    // PROFILE_END(profile_name + "::shape_cuda_value");
    // cudaEventRecord(stop);
    // cudaEventSynchronize(stop);
    // cudaEventElapsedTime(&milliseconds, start, stop);
    // printf("shape_cuda of v_proj: %.2f ms\n", milliseconds * this->num_heads);

    int start_idx = 0;
    if (input.has_past_key_value) start_idx = input.past_key.m_dim_y;
    // printf("dddddddddd\n");

    // cudaEventRecord(start);
    // PROFILE_START(profile_name + "::RotaryPosEmb_cuda_forward");
    dim3 grid(num_heads, 1, 1);
    dim3 block(sqlen, 1, 1);
    RotaryPosEmb_cuda_forward<<<grid, block>>>(query_states, key_states, this->rotary_pos_emb.cos, this->rotary_pos_emb.sin, start_idx, sqlen);
    // PROFILE_END(profile_name + "::RotaryPosEmb_cuda_forward");
    // cudaEventRecord(stop);
    // cudaEventSynchronize(stop);
    // cudaEventElapsedTime(&milliseconds, start, stop);
    // printf("RotaryPosEmb_cuda_forward: %.2f ms\n", milliseconds * this->num_heads);

    // cudaEventRecord(start);
    // PROFILE_START(profile_name + "::cat_past_keys_values");
    int tgz = sqlen;
    if (input.has_past_key_value) {
        assert(input.past_key.m_dim_z == this->head_dim);
        tgz += input.past_key.m_dim_y;
        float16_t *val_ptr = ret_value_states, *key_ptr = ret_key_states;
        int past_block = input.past_key.m_dim_y * input.past_key.m_dim_z;
        int sq_block = sqlen * this->head_dim;
        for (int i = 0; i < input.past_key.m_dim_x; i++) {
            cudaMemcpyAsync(val_ptr, &input.past_value.m_data[past_block * i], past_block * sizeof(float16_t), cudaMemcpyDeviceToDevice);
            val_ptr += past_block;
            cudaMemcpyAsync(val_ptr, &value_states.m_data[sq_block * i], sq_block * sizeof(float16_t), cudaMemcpyDeviceToDevice);
            val_ptr += sq_block;
            cudaMemcpyAsync(key_ptr, &input.past_key.m_data[past_block * i], past_block * sizeof(float16_t), cudaMemcpyDeviceToDevice);
            key_ptr += past_block;
            cudaMemcpyAsync(key_ptr, &key_states.m_data[sq_block * i], sq_block * sizeof(float16_t), cudaMemcpyDeviceToDevice);
            key_ptr += sq_block;
        }
    } else {
        cudaMemcpyAsync(ret_value_states, value_states_arr, (this->num_heads * tgz * this->head_dim) * sizeof(float16_t), cudaMemcpyDeviceToDevice);
        cudaMemcpyAsync(ret_key_states, key_states_arr, (this->num_heads * tgz * this->head_dim) * sizeof(float16_t), cudaMemcpyDeviceToDevice);
    }

    // printf("eeeeeeeeee\n");

    Matrix3D<float16_t> final_value_states(ret_value_states, this->num_heads, tgz, this->head_dim);
    Matrix3D<float16_t> final_key_states(ret_key_states, this->num_heads, tgz, this->head_dim);
    // PROFILE_END(profile_name + "::cat_past_keys_values");
    // cudaEventRecord(stop);
    // cudaEventSynchronize(stop);
    // cudaEventElapsedTime(&milliseconds, start, stop);
    // printf("cat_past_keys_values: %.2f ms\n", milliseconds * this->num_heads);

    // cudaEventRecord(start);
    // PROFILE_START(profile_name + "::qk_bmm");
    Matrix3D<float16_t> attn_weights(attn_weights_arr, this->num_heads, sqlen, tgz);
    this->qk_bmm.forward(query_states, final_key_states, attn_weights);
    // PROFILE_END(profile_name + "::qk_bmm");
    // cudaEventRecord(stop);
    // cudaEventSynchronize(stop);
    // cudaEventElapsedTime(&milliseconds, start, stop);
    // printf("qk_bmm.forward: %.2f ms\n", milliseconds * this->num_heads);

    // cudaEventRecord(start);
    // PROFILE_START(profile_name + "::batch_Add_cuda");
    dim3 threadsPerBlock2(16, 4, 16);
    dim3 numBlocks2((this->num_heads + threadsPerBlock2.x - 1) / threadsPerBlock2.x,
                (sqlen + threadsPerBlock2.y - 1) / threadsPerBlock2.y,
                (tgz + threadsPerBlock2.z - 1) / threadsPerBlock2.z);
    // if (attn_weights.m_dim_z % 2 == 0) {
    //     batch_Add_cuda_half2<<<numBlocks2, threadsPerBlock2>>>(attn_weights, input.attention_mask, attn_weights);
    // } else {
    //     batch_Add_cuda<<<numBlocks2, threadsPerBlock2>>>(attn_weights, input.attention_mask, attn_weights);
    // }
    batch_Add_cuda<<<numBlocks2, threadsPerBlock2>>>(attn_weights, input.attention_mask, attn_weights);
    // PROFILE_END(profile_name + "::batch_Add_cuda");
    // cudaEventRecord(stop);
    // cudaEventSynchronize(stop);
    // cudaEventElapsedTime(&milliseconds, start, stop);
    // printf("batch_Add_cuda: %.2f ms\n", milliseconds * this->num_heads);
    // printf("fffffffff\n");

    // cudaEventRecord(start);
    // PROFILE_START(profile_name + "::check_inf_half");
    int threadsPerBlock_1D = 1024;
    int blocksPerGrid_1D =(attn_weights.length() + threadsPerBlock_1D - 1) / threadsPerBlock_1D;
    check_inf_half<<<blocksPerGrid_1D, threadsPerBlock_1D>>>(attn_weights);
    // PROFILE_END(profile_name + "::check_inf_half");
    // cudaEventRecord(stop);
    // cudaEventSynchronize(stop);
    // cudaEventElapsedTime(&milliseconds, start, stop);
    // printf("check_inf_half: %.2f ms\n", milliseconds * this->num_heads);

    // cudaEventRecord(start);
    // PROFILE_START(profile_name + "::softmax_cuda");
    Matrix3D<float16_t> attn_probs(attn_weights_arr, this->num_heads, sqlen, tgz);
    dim3 threadsPerBlock3(64, 16);
    dim3 numBlocks3((this->num_heads + threadsPerBlock3.x - 1) / threadsPerBlock3.x, (sqlen + threadsPerBlock3.y - 1) / threadsPerBlock3.y);
    softmax_cuda<<<numBlocks3, threadsPerBlock3>>>(attn_weights, attn_probs);
    // PROFILE_END(profile_name + "::softmax_cuda");
    // cudaEventRecord(stop);
    // cudaEventSynchronize(stop);
    // cudaEventElapsedTime(&milliseconds, start, stop);
    // printf("softmax_cuda: %.2f ms\n", milliseconds * this->num_heads);
    // printf("gggggggggg\n");


    /* Legacy Implementation of PV_BMM*/
    // cudaEventRecord(start);
    // PROFILE_START(profile_name + "::transpose_1_2idx_cuda");
    Matrix3D<float16_t> value_states_transpose(value_states_transpose_arr, this->num_heads, this->head_dim, tgz);
    dim3 threadsPerBlock4(8, 4, 32);
    dim3 numBlocks4((this->num_heads + threadsPerBlock4.x - 1) / threadsPerBlock4.x,
                (tgz + threadsPerBlock4.y - 1) / threadsPerBlock4.y,
                (this->head_dim + threadsPerBlock4.z - 1) / threadsPerBlock4.z);
    transpose_1_2idx_cuda<<<numBlocks4, threadsPerBlock4>>>(final_value_states, value_states_transpose);
    // PROFILE_END(profile_name + "::transpose_1_2idx_cuda");
    // cudaEventRecord(stop);
    // cudaEventSynchronize(stop);
    // cudaEventElapsedTime(&milliseconds, start, stop);
    // printf("transpose_1_2idx_cuda: %.2f ms\n", milliseconds * this->num_heads);

    // cudaEventRecord(start);
    // PROFILE_START(profile_name + "::pv_bmm");
    Matrix3D<float16_t> attn_output(attn_output_arr, this->num_heads, sqlen, this->head_dim);
    this->pv_bmm.forward(attn_probs, value_states_transpose, attn_output);
    // PROFILE_END(profile_name + "::pv_bmm");
    // cudaEventRecord(stop);
    // cudaEventSynchronize(stop);
    // cudaEventElapsedTime(&milliseconds, start, stop);
    // printf("pv_bmm.forward: %.2f ms\n", milliseconds * this->num_heads);

    // /* Optimized Implementation of PV_BMM*/
    // // cudaEventRecord(start);
    // // PROFILE_START(profile_name + "::pv_bmm");
    // Matrix3D<float16_t> attn_output(attn_output_arr, this->num_heads, sqlen, this->head_dim);
    // this->pv_bmm.forward_weight_untransposed(attn_probs, final_value_states, attn_output);
    // // PROFILE_END(profile_name + "::pv_bmm");
    // // cudaEventRecord(stop);
    // // cudaEventSynchronize(stop);
    // // cudaEventElapsedTime(&milliseconds, start, stop);
    // // printf("pv_bmm.forward: %.2f ms\n", milliseconds * this->num_heads);
    // printf("hhhhhhhhhh\n");


    // cudaEventRecord(start);
    // PROFILE_START(profile_name + "::unshape_cuda");
    Matrix3D<float16_t> attn_output_transpose(attn_output_transpose_arr, 1, sqlen, this->num_heads * this->head_dim);
    unshape_cuda<<<numBlocks, threadsPerBlock>>>(attn_output, attn_output_transpose, this->num_heads, sqlen, this->head_dim);
    // PROFILE_END(profile_name + "::unshape_cuda");
    // cudaEventRecord(stop);
    // cudaEventSynchronize(stop);
    // cudaEventElapsedTime(&milliseconds, start, stop);
    // printf("unshape_cuda: %.2f ms\n", milliseconds * this->num_heads);

    // cudaEventRecord(start);
    // PROFILE_START(profile_name + "::o_proj.forward");
    Matrix3D<float16_t> attn_output_half(attn_output_half_arr, 1, sqlen, this->num_heads * this->head_dim);
    this->o_proj.forward(attn_output_transpose, attn_output_half, split_8_buffer);
    // PROFILE_END(profile_name + "::o_proj.forward");
    // cudaEventRecord(stop);
    // cudaEventSynchronize(stop);
    // cudaEventElapsedTime(&milliseconds, start, stop);
    // printf("o_proj.forward: %.2f ms\n", milliseconds * this->num_heads);

    // output assignment
    output.attn_output = attn_output_half;
    output.past_key_value = {final_key_states, final_value_states};
    
    // cudaEventRecord(stop_Attention);
    // cudaEventSynchronize(stop_Attention);
    // cudaEventElapsedTime(&milliseconds, start_Attention, stop_Attention);
    // printf("The elapsed time of Int4llamaAttention is %.2f ms\n", milliseconds * this->num_heads);

    PROFILE_END(profile_name);

    // printf("iiiiiiiiii\n");

    // cudaEventDestroy(start_Attention);
    // cudaEventDestroy(stop_Attention);
    // cudaEventDestroy(start);
    // cudaEventDestroy(stop);

    return output;
}

void Int4llamaAttention::free_cuda_memory() {
    free_aligned_memory_gpu(attn_weights_arr);
    free_aligned_memory_gpu(attn_output_half_arr);
    free_aligned_memory_gpu(query_states_unshape_arr);
    free_aligned_memory_gpu(attn_output_arr);
    free_aligned_memory_gpu(attn_output_transpose_arr);
    free_aligned_memory_gpu(key_states_unshape_arr);
    free_aligned_memory_gpu(key_states_arr);
    free_aligned_memory_gpu(value_states_unshape_arr);
    free_aligned_memory_gpu(value_states_arr);
    free_aligned_memory_gpu(query_states_arr);
    free_aligned_memory_gpu(value_states_transpose_arr);
    free_aligned_memory_gpu(key_states_arr_cache);
    free_aligned_memory_gpu(value_states_arr_cache);
    free_aligned_memory_gpu(cos_buf);
    free_aligned_memory_gpu(sin_buf);
    free_aligned_memory_gpu(q_weight);
    free_aligned_memory_gpu(k_weight);
    free_aligned_memory_gpu(v_weight);
    free_aligned_memory_gpu(o_weight);

    if(cache_num) {
        free(cache_num);
        cache_num = nullptr;
    }
}
