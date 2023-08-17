#include "Int4llamaDecoderLayer.h"
#include "utils.h"

// Shared memory space across all layers
static float16_t *hidden_states_half_arr = nullptr;
static float16_t *final_layer_norm_arr = nullptr;
static float16_t *gate_proj_arr = nullptr;
static float16_t *up_proj_arr = nullptr;
static float16_t *down_proj_arr = nullptr;
static float16_t *hidden_states_arr = nullptr;

__global__ void add_half(Matrix3D<float16_t> a, Matrix3D<float16_t> b, Matrix3D<float16_t> c) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (i < a.length()) {
        c.m_data[i] = __hadd(a.m_data[i], b.m_data[i]);
    }
}

// __global__ void add_float(Matrix3D<float> a, Matrix3D<float> b, Matrix3D<float> c) {
//     int i = threadIdx.x + blockIdx.x * blockDim.x;
    
//     if (i < a.length()) {
//         c.m_data[i] = a.m_data[i] + b.m_data[i];
//     }
// }

__global__ void SiLuMul_half(Matrix3D<float16_t> a, Matrix3D<float16_t> b) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    half scalar_one = 1;

    if (i < a.length()) {
        float16_t v = a.m_data[i];
        float16_t silu_v = __hmul(v, __hdiv(scalar_one, __hadd(scalar_one, hexp(__hneg(v)))));
        a.m_data[i] = __hmul(silu_v, b.m_data[i]);
    }
}

template<typename T>
__device__ __forceinline__ T silu(const T& x) {
  // x * sigmoid(x)
  return (T) (((float) x) / (1.0f + expf((float) -x)));
}

template<typename scalar_t>
__global__ void silu_and_mul_kernel(
  scalar_t* __restrict__ out,        
  const scalar_t* __restrict__ input, 
  const int d) {
  const int token_idx = blockIdx.x;
  for (int idx = threadIdx.x; idx < d; idx += blockDim.x) {
    const scalar_t x = __ldg(&out[token_idx * d + idx]);
    const scalar_t y = __ldg(&input[token_idx * d + idx]);
    out[token_idx * d + idx] = silu(x) * y;
  }
}

// __global__ void SiLuMul_float(Matrix3D<float> a, Matrix3D<float> b) {
//     int i = threadIdx.x + blockIdx.x * blockDim.x;

//     if (i < a.length()) {
//         float v = a.m_data[i];
//         float silu_v = v * (1.0 / (1.0 + __expf(-1.0 * v)));
//         a.m_data[i] = silu_v * b.m_data[i];
//     }
// }

Int4llamaDecoderLayer::Int4llamaDecoderLayer(std::string param_path, const struct model_config config, int layer_idx) {
    if (layer_idx == 0) {
        allocate_aligned_memory_gpu(hidden_states_half_arr, config.max_sqlen * config.embed_dim * sizeof(float16_t));
        allocate_aligned_memory_gpu(final_layer_norm_arr, config.max_sqlen * config.embed_dim * sizeof(float16_t));
        allocate_aligned_memory_gpu(gate_proj_arr, config.max_sqlen * config.hidden_dim * sizeof(float16_t));
        allocate_aligned_memory_gpu(up_proj_arr, config.max_sqlen * config.hidden_dim * sizeof(float16_t));
        allocate_aligned_memory_gpu(down_proj_arr, config.max_sqlen * config.embed_dim * sizeof(float16_t));
        allocate_aligned_memory_gpu(hidden_states_arr, config.max_sqlen * config.embed_dim * sizeof(float16_t));
        Int4llamaAttention::initialized_memory(config);
        // Int4llamaAttention llamaAttention;
        // llamaAttention.initialized_memory(config);

        // allocate_aligned_memory_gpu(split_8_buffer, config.max_sqlen * config.hidden_dim * sizeof(float16_t) * 8);
        // printf("config.max_sqlen, config.hidden_dim: %d, %d\n", config.max_sqlen, config.hidden_dim);
    }

    allocate_aligned_memory_gpu(input_layernorm_weight_ptr, config.embed_dim * sizeof(float));
    Matrix3D<float> input_layernorm_weight(input_layernorm_weight_ptr, 1, 1, config.embed_dim);
    input_layernorm_weight.load((param_path + "/input_layernorm/weight.bin").c_str());
    this->input_layernorm = LlamaRMSNorm_cuda(input_layernorm_weight);

    allocate_aligned_memory_gpu(post_attention_layernorm_ptr, config.embed_dim * sizeof(float));
    Matrix3D<float> post_attention_layernorm_weight(post_attention_layernorm_ptr, 1, 1, config.embed_dim);
    post_attention_layernorm_weight.load((param_path + "/post_attention_layernorm/weight.bin").c_str());
    this->post_attention_layernorm = LlamaRMSNorm_cuda(post_attention_layernorm_weight);

    this->embed_dim = config.embed_dim;
    this->num_attention_heads = config.num_heads;
    this->hidden_dim = config.hidden_dim;
    this->layer_idx = layer_idx;

    this->attn = Int4llamaAttention(param_path + "/self_attn", config);

    allocate_aligned_memory_gpu(gate_proj_weight, (config.embed_dim * config.hidden_dim * sizeof(int)) / 8);
    allocate_aligned_memory_gpu(down_proj_weight, (config.hidden_dim * config.embed_dim * sizeof(int)) / 8);
    allocate_aligned_memory_gpu(up_proj_weight, (config.embed_dim * config.hidden_dim * sizeof(int)) / 8);
    this->gate_proj = Linear_half_int4(Matrix3D<int>(gate_proj_weight, 1, config.hidden_dim / 8, config.embed_dim),
                                     (param_path + "/gate_proj"));
    this->down_proj = Linear_half_int4(Matrix3D<int>(down_proj_weight, 1, config.embed_dim / 8, config.hidden_dim),
                                     (param_path + "/down_proj"));
    this->up_proj = Linear_half_int4(Matrix3D<int>(up_proj_weight, 1, config.hidden_dim / 8, config.embed_dim),
                                   (param_path + "/up_proj"));
}

struct Int4llamaDecoderLayer_output Int4llamaDecoderLayer::forward(const struct Int4llamaDecoderLayer_input &input) {
    PROFILE_START(profile_name);
    
    // cudaEvent_t start, stop;
    // cudaEventCreate(&start);
    // cudaEventCreate(&stop);
    // float milliseconds = 0;


    // cudaEventRecord(start);

    Matrix3D<float16_t> hidden_states(hidden_states_arr, input.hidden_states.m_dim_x, input.hidden_states.m_dim_y,
                                  input.hidden_states.m_dim_z);
    this->input_layernorm.forward(input.hidden_states, hidden_states);

    // cudaEventRecord(stop);
    // cudaEventSynchronize(stop);
    // cudaEventElapsedTime(&milliseconds, start, stop);
    // printf("input_layernorm.forward: %.2f ms\n", milliseconds * this->num_attention_heads);


    // cudaEventRecord(start);

    struct Int4llamaAttention_input attn_param(hidden_states, input.attention_mask, input.past_key, input.past_value,
                                               input.has_past_key_value, this->layer_idx);
    struct Int4llamaAttention_output attn_output = this->attn.forward(attn_param);

    // cudaEventRecord(stop);
    // cudaEventSynchronize(stop);
    // cudaEventElapsedTime(&milliseconds, start, stop);
    // printf("this->attn.forward: %.2f ms\n", milliseconds * this->num_attention_heads);


    // cudaEventRecord(start);

    Matrix3D<float16_t> residual_add(hidden_states_half_arr, input.hidden_states.m_dim_x, input.hidden_states.m_dim_y,
                                 input.hidden_states.m_dim_z);
    int threadsPerBlock = 1024;
    int blocksPerGrid =(input.hidden_states.length() + threadsPerBlock - 1) / threadsPerBlock;
    add_half<<<blocksPerGrid, threadsPerBlock>>>(input.hidden_states, attn_output.attn_output, residual_add);

    // cudaEventRecord(stop);
    // cudaEventSynchronize(stop);
    // cudaEventElapsedTime(&milliseconds, start, stop);
    // printf("add_half: %.2f ms\n", milliseconds * this->num_attention_heads);


    // cudaEventRecord(start);

    Matrix3D<float16_t> post_attention_layernorm(final_layer_norm_arr, input.hidden_states.m_dim_x,
                                             input.hidden_states.m_dim_y, input.hidden_states.m_dim_z);
    this->post_attention_layernorm.forward(residual_add, post_attention_layernorm);

    // cudaEventRecord(stop);
    // cudaEventSynchronize(stop);
    // cudaEventElapsedTime(&milliseconds, start, stop);
    // printf("post_attention_layernorm.forward: %.2f ms\n", milliseconds * this->num_attention_heads);


    // cudaEventRecord(start);

    Matrix3D<float16_t> gate_proj(gate_proj_arr, input.hidden_states.m_dim_x, input.hidden_states.m_dim_y,
                              this->hidden_dim);
    this->gate_proj.forward(post_attention_layernorm, gate_proj, split_8_buffer);

    // cudaEventRecord(stop);
    // cudaEventSynchronize(stop);
    // cudaEventElapsedTime(&milliseconds, start, stop);
    // printf("gate_proj.forward: %.2f ms\n", milliseconds * this->num_attention_heads);


    // cudaEventRecord(start);

    Matrix3D<float16_t> up_proj(up_proj_arr, input.hidden_states.m_dim_x, input.hidden_states.m_dim_y, this->hidden_dim);
    this->up_proj.forward(post_attention_layernorm, up_proj, split_8_buffer);

    // cudaEventRecord(stop);
    // cudaEventSynchronize(stop);
    // cudaEventElapsedTime(&milliseconds, start, stop);
    // printf("up_proj.forward: %.2f ms\n", milliseconds * this->num_attention_heads);


    // cudaEventRecord(start);

    int blocksPerGrid2 =(gate_proj.length() + threadsPerBlock - 1) / threadsPerBlock;
    SiLuMul_half<<<blocksPerGrid2, threadsPerBlock>>>(gate_proj, up_proj);
    // dim3 grid(input.hidden_states.m_dim_x * input.hidden_states.m_dim_y);
    // dim3 block(std::min(this->hidden_dim, 1024));
    // silu_and_mul_kernel<<<grid, block>>>(gate_proj.m_data, up_proj.m_data, this->hidden_dim);

    // cudaEventRecord(stop);
    // cudaEventSynchronize(stop);
    // cudaEventElapsedTime(&milliseconds, start, stop);
    // printf("SiLuMul_half: %.2f ms\n", milliseconds * this->num_attention_heads);


    // cudaEventRecord(start);

    Matrix3D<float16_t> down_proj(down_proj_arr, input.hidden_states.m_dim_x, input.hidden_states.m_dim_y, this->embed_dim);
    this->down_proj.forward(gate_proj, down_proj, split_8_buffer);

    // cudaEventRecord(stop);
    // cudaEventSynchronize(stop);
    // cudaEventElapsedTime(&milliseconds, start, stop);
    // printf("down_proj.forward: %.2f ms\n", milliseconds * this->num_attention_heads);


    // cudaEventRecord(start);

    int blocksPerGrid3 =(residual_add.length() + threadsPerBlock - 1) / threadsPerBlock;
    add_half<<<blocksPerGrid3, threadsPerBlock>>>(residual_add, down_proj, residual_add);

    // cudaEventRecord(stop);
    // cudaEventSynchronize(stop);
    // cudaEventElapsedTime(&milliseconds, start, stop);
    // printf("add_half: %.2f ms\n", milliseconds * this->num_attention_heads);


    struct Int4llamaDecoderLayer_output output(residual_add, attn_output.attn_probs_reshaped,
                                               attn_output.past_key_value);
    PROFILE_END(profile_name);

    // cudaEventDestroy(start);
    // cudaEventDestroy(stop);

    return output;
}

void Int4llamaDecoderLayer::free_cuda_memory() {
    free_aligned_memory_gpu(hidden_states_half_arr);
    free_aligned_memory_gpu(final_layer_norm_arr);
    free_aligned_memory_gpu(gate_proj_arr);
    free_aligned_memory_gpu(up_proj_arr);
    free_aligned_memory_gpu(down_proj_arr);
    free_aligned_memory_gpu(hidden_states_arr);
    free_aligned_memory_gpu(input_layernorm_weight_ptr);
    free_aligned_memory_gpu(post_attention_layernorm_ptr);
    free_aligned_memory_gpu(gate_proj_weight);
    free_aligned_memory_gpu(down_proj_weight);
    free_aligned_memory_gpu(up_proj_weight);
}