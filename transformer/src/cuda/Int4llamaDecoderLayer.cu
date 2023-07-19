#include "Int4llamaDecoderLayer.h"

#include "utils.h"

// Shared memory space across all layers
static float *hidden_states_float_arr;
static float *final_layer_norm_arr;
static float *gate_proj_arr;
static float *up_proj_arr;
static float *down_proj_arr;
static float *temp;
static float *hidden_states_arr;

template <typename T>
static void add(Matrix3D<T> a, Matrix3D<T> b, Matrix3D<T> c) {
    PROFILE_START("Int4llamaDecoderLayer::add");
    assert(c.length() == a.length() && a.length() == b.length());

    for (int i = 0; i < a.length(); i++) {
        c.m_data[i] = a.m_data[i] + b.m_data[i];
    }
    PROFILE_END("Int4llamaDecoderLayer::add");
}

__global__ void add_half(Matrix3D<float> a, Matrix3D<float> b, Matrix3D<float> c) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (i < a.length()) {
        c.m_data[i] = a.m_data[i] + b.m_data[i];
    }
}

static void SiLuMul(Matrix3D<float> a, Matrix3D<float> b) {
    PROFILE_START("Int4llamaDecoderLayer::MulSiLu");
    for (int i = 0; i < a.length(); i++) {
        float v = a.m_data[i];
        float silu_v = v * (1.0 / (1.0 + exp(-1 * v)));
        a.m_data[i] = silu_v * b.m_data[i];
    }
    PROFILE_END("Int4llamaDecoderLayer::MulSiLu");
}

__global__ void SiLuMul_half(Matrix3D<float> a, Matrix3D<float> b) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    if (i < a.length()) {
        float v = a.m_data[i];
        float silu_v = v * (1.0 / (1.0 + __expf(-1.0 * v)));
        a.m_data[i] = silu_v * b.m_data[i];
    }
}


Int4llamaDecoderLayer::Int4llamaDecoderLayer(std::string param_path, const struct model_config config, int layer_idx) {
    if (layer_idx == 0) {
        allocate_aligned_memory_gpu(hidden_states_float_arr, config.max_sqlen * config.embed_dim * sizeof(float));
        allocate_aligned_memory_gpu(final_layer_norm_arr, config.max_sqlen * config.embed_dim * sizeof(float));
        allocate_aligned_memory_gpu(gate_proj_arr, config.max_sqlen * config.hidden_dim * sizeof(float));
        allocate_aligned_memory_gpu(up_proj_arr, config.max_sqlen * config.hidden_dim * sizeof(float));
        allocate_aligned_memory_gpu(down_proj_arr, config.max_sqlen * config.embed_dim * sizeof(float));
        allocate_aligned_memory_gpu(hidden_states_arr, config.max_sqlen * config.embed_dim * sizeof(float));
        Int4llamaAttention::initialized_memory(config);
    }

    float *input_layernorm_weight_ptr;
    allocate_aligned_memory_gpu(input_layernorm_weight_ptr, config.embed_dim * sizeof(float));
    Matrix3D<float> input_layernorm_weight(input_layernorm_weight_ptr, 1, 1, config.embed_dim);
    input_layernorm_weight.load((param_path + "/input_layernorm/weight.bin").c_str());
    //// Original code
    // this->input_layernorm = LlamaRMSNorm(input_layernorm_weight);
    //// CUDA 1
    this->input_layernorm = LlamaRMSNorm_half(input_layernorm_weight);

    float *post_attention_layernorm_ptr;
    allocate_aligned_memory_gpu(post_attention_layernorm_ptr, config.embed_dim * sizeof(float));
    Matrix3D<float> post_attention_layernorm_weight(post_attention_layernorm_ptr, 1, 1, config.embed_dim);
    post_attention_layernorm_weight.load((param_path + "/post_attention_layernorm/weight.bin").c_str());
    //// Original code
    // this->post_attention_layernorm = LlamaRMSNorm(post_attention_layernorm_weight);
    //// CUDA 1
    this->post_attention_layernorm = LlamaRMSNorm_half(post_attention_layernorm_weight);

    this->embed_dim = config.embed_dim;
    this->num_attention_heads = config.num_heads;
    this->hidden_dim = config.hidden_dim;
    this->layer_idx = layer_idx;

    this->attn = Int4llamaAttention(param_path + "/self_attn", config);

    allocate_aligned_memory_gpu(gate_proj_weight, (config.embed_dim * config.hidden_dim * sizeof(int)) / 8);
    allocate_aligned_memory_gpu(down_proj_weight, (config.hidden_dim * config.embed_dim * sizeof(int)) / 8);
    allocate_aligned_memory_gpu(up_proj_weight, (config.embed_dim * config.hidden_dim * sizeof(int)) / 8);
    this->gate_proj = Linear_half_int4_ref(Matrix3D<int>(gate_proj_weight, 1, config.hidden_dim / 8, config.embed_dim),
                                     (param_path + "/gate_proj"));
    this->down_proj = Linear_half_int4_ref(Matrix3D<int>(down_proj_weight, 1, config.embed_dim / 8, config.hidden_dim),
                                     (param_path + "/down_proj"));
    this->up_proj = Linear_half_int4_ref(Matrix3D<int>(up_proj_weight, 1, config.hidden_dim / 8, config.embed_dim),
                                   (param_path + "/up_proj"));
}


struct Int4llamaDecoderLayer_output Int4llamaDecoderLayer::forward(const struct Int4llamaDecoderLayer_input &input) {
    PROFILE_START(profile_name);
    Matrix3D<float> hidden_states(hidden_states_arr, input.hidden_states.m_dim_x, input.hidden_states.m_dim_y,
                                  input.hidden_states.m_dim_z);
    this->input_layernorm.forward(input.hidden_states, hidden_states);
    // cudaDeviceSynchronize();

    struct Int4llamaAttention_input attn_param(hidden_states, input.attention_mask, input.past_key, input.past_value,
                                               input.has_past_key_value, this->layer_idx);
    struct Int4llamaAttention_output attn_output = this->attn.forward(attn_param);

    Matrix3D<float> residual_add(hidden_states_float_arr, input.hidden_states.m_dim_x, input.hidden_states.m_dim_y,
                                 input.hidden_states.m_dim_z);
    //// Original code
    // add(input.hidden_states, attn_output.attn_output, residual_add);
    //// CUDA 1
    int threadsPerBlock = 1024;
    int blocksPerGrid =(input.hidden_states.length() + threadsPerBlock - 1) / threadsPerBlock;
    add_half<<<blocksPerGrid, threadsPerBlock>>>(input.hidden_states, attn_output.attn_output, residual_add);
    // cudaDeviceSynchronize();
    // cudaError_t err;
    // err = cudaGetLastError();
    // if (err != cudaSuccess) {
    // printf("Error launching add_half 1 kernel: %s\n", cudaGetErrorString(err));
    // }

    Matrix3D<float> post_attention_layernorm(final_layer_norm_arr, input.hidden_states.m_dim_x,
                                             input.hidden_states.m_dim_y, input.hidden_states.m_dim_z);
    this->post_attention_layernorm.forward(residual_add, post_attention_layernorm);
    // cudaDeviceSynchronize();
    // err = cudaGetLastError();
    // if (err != cudaSuccess) {
    // printf("Error launching post_attention_layernorm kernel: %s\n", cudaGetErrorString(err));
    // }

    Matrix3D<float> gate_proj(gate_proj_arr, input.hidden_states.m_dim_x, input.hidden_states.m_dim_y,
                              this->hidden_dim);
    this->gate_proj.forward(post_attention_layernorm, gate_proj);

    Matrix3D<float> up_proj(up_proj_arr, input.hidden_states.m_dim_x, input.hidden_states.m_dim_y, this->hidden_dim);
    this->up_proj.forward(post_attention_layernorm, up_proj);

    //// Original code
    // SiLuMul(gate_proj, up_proj);
    //// CUDA 1
    // int threadsPerBlock = 256;
    int blocksPerGrid2 =(gate_proj.length() + threadsPerBlock - 1) / threadsPerBlock;
    SiLuMul_half<<<blocksPerGrid2, threadsPerBlock>>>(gate_proj, up_proj);
    // cudaDeviceSynchronize();

    Matrix3D<float> down_proj(down_proj_arr, input.hidden_states.m_dim_x, input.hidden_states.m_dim_y, this->embed_dim);
    this->down_proj.forward(gate_proj, down_proj);

    //// Original code
    // add(residual_add, down_proj, residual_add);
    //// CUDA 1
    int blocksPerGrid3 =(residual_add.length() + threadsPerBlock - 1) / threadsPerBlock;
    add_half<<<blocksPerGrid3, threadsPerBlock>>>(residual_add, down_proj, residual_add);
    // cudaDeviceSynchronize();

    struct Int4llamaDecoderLayer_output output(residual_add, attn_output.attn_probs_reshaped,
                                               attn_output.past_key_value);
    PROFILE_END(profile_name);

    // cudaFree(gate_proj_weight);
    // cudaFree(down_proj_weight);
    // cudaFree(up_proj_weight);

    return output;
}
