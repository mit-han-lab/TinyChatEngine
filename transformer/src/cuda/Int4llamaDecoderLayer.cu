#include "Int4llamaDecoderLayer.h"
#include "utils.h"
#include "utils.cuh"

// Shared memory space across all layers
static float16_t *hidden_states_half_arr;
static float16_t *final_layer_norm_arr;
static float16_t *gate_proj_arr;
static float16_t *up_proj_arr;
static float16_t *down_proj_arr;
static float16_t *hidden_states_arr;

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

    if (i < a.length()) {
        float16_t v = a.m_data[i];
        float16_t silu_v = __hmul(v, __hdiv(__float2half(1.0f), __hadd(__float2half(1.0f), hexp(__hneg(v)))));
        a.m_data[i] = __hmul(silu_v, b.m_data[i]);
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
    }

    float *input_layernorm_weight_ptr;
    allocate_aligned_memory_gpu(input_layernorm_weight_ptr, config.embed_dim * sizeof(float));
    Matrix3D<float> input_layernorm_weight(input_layernorm_weight_ptr, 1, 1, config.embed_dim);
    input_layernorm_weight.load((param_path + "/input_layernorm/weight.bin").c_str());
    this->input_layernorm = LlamaRMSNorm_cuda(input_layernorm_weight);

    float *post_attention_layernorm_ptr;
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
    Matrix3D<float16_t> hidden_states(hidden_states_arr, input.hidden_states.m_dim_x, input.hidden_states.m_dim_y,
                                  input.hidden_states.m_dim_z);
    this->input_layernorm.forward(input.hidden_states, hidden_states);

    struct Int4llamaAttention_input attn_param(hidden_states, input.attention_mask, input.past_key, input.past_value,
                                               input.has_past_key_value, this->layer_idx);
    struct Int4llamaAttention_output attn_output = this->attn.forward(attn_param);

    Matrix3D<float16_t> residual_add(hidden_states_half_arr, input.hidden_states.m_dim_x, input.hidden_states.m_dim_y,
                                 input.hidden_states.m_dim_z);
    int threadsPerBlock = 1024;
    int blocksPerGrid =(input.hidden_states.length() + threadsPerBlock - 1) / threadsPerBlock;
    add_half<<<blocksPerGrid, threadsPerBlock>>>(input.hidden_states, attn_output.attn_output, residual_add);

    Matrix3D<float16_t> post_attention_layernorm(final_layer_norm_arr, input.hidden_states.m_dim_x,
                                             input.hidden_states.m_dim_y, input.hidden_states.m_dim_z);
    this->post_attention_layernorm.forward(residual_add, post_attention_layernorm);

    Matrix3D<float16_t> gate_proj(gate_proj_arr, input.hidden_states.m_dim_x, input.hidden_states.m_dim_y,
                              this->hidden_dim);
    this->gate_proj.forward(post_attention_layernorm, gate_proj);

    Matrix3D<float16_t> up_proj(up_proj_arr, input.hidden_states.m_dim_x, input.hidden_states.m_dim_y, this->hidden_dim);
    this->up_proj.forward(post_attention_layernorm, up_proj);

    int blocksPerGrid2 =(gate_proj.length() + threadsPerBlock - 1) / threadsPerBlock;
    SiLuMul_half<<<blocksPerGrid2, threadsPerBlock>>>(gate_proj, up_proj);

    Matrix3D<float16_t> down_proj(down_proj_arr, input.hidden_states.m_dim_x, input.hidden_states.m_dim_y, this->embed_dim);
    this->down_proj.forward(gate_proj, down_proj);

    int blocksPerGrid3 =(residual_add.length() + threadsPerBlock - 1) / threadsPerBlock;
    add_half<<<blocksPerGrid3, threadsPerBlock>>>(residual_add, down_proj, residual_add);

    struct Int4llamaDecoderLayer_output output(residual_add, attn_output.attn_probs_reshaped,
                                               attn_output.past_key_value);
    PROFILE_END(profile_name);

    return output;
}
