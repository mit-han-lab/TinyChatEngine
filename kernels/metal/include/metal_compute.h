#ifndef METAL_COMPUTE_H
#define METAL_COMPUTE_H

#include "../../matmul.h"
#include "operators.h"
#include "Foundation/Foundation.hpp"
#include "Metal/Metal.hpp"

bool has_init = false;

struct metal_kernel {
    MTL::ComputePipelineState * pipeline;
};

struct metal_context * ctx;
struct metal_cgraph * mgraph;

enum {
    MTLGPUFamilyApple1 = 1001, 
    MTLGPUFamilyCommon1 = 3001,
    MTLGPUFamilyMetal3 = 5001,
    MTLGPUFamilyApple7 = 1007,
};

enum metal_kernel_type {
    METAL_KERNEL_FLOAT2HALF,
    METAL_KERNEL_HALF2FLOAT,
    METAL_KERNEL_PREPARE_DECODER_ATTENTION_MASK_HALF,
    METAL_KERNEL_SILUMUL_HALF,
    METAL_KERNEL_ADD_HALF,
    METAL_KERNEL_SHAPE_QKV,
    METAL_KERNEL_UNSHAPE,
    METAL_KERNEL_TRANSPOSE_1_2IDX,
    METAL_KERNEL_CHECK_INF_HALF,
    METAL_KERNEL_EMBEDDING,
    METAL_KERNEL_BATCH_ADD,
    METAL_KERNEL_RELU,
    METAL_KERNEL_SILU,
    METAL_KERNEL_GELU,
    METAL_KERNEL_GELU_QUICK,
    METAL_KERNEL_RMS_NORM,
    METAL_KERNEL_SOFT_MAX,
    METAL_KERNEL_SOFT_MAX_4,
    METAL_KERNEL_ROPE,
    METAL_KERNEL_MUL_MM_INT4_F32,
    METAL_KERNEL_MUL_MV_INT4_F32,
    METAL_KERNEL_MUL_MM_F32_F32,
    METAL_KERNEL_MUL_MV_F32_F32,
    METAL_KERNEL_TYPE_COUNT
};

enum status {
    STATUS_SUCCESS,
    STATUS_FAILED
};

// Context struct holding Metal related objects and state
struct metal_context {
    int n_cb;
    MTL::Device * device;
    MTL::CommandQueue * queue;
    static std::unordered_map<void *, MTL::Buffer *> _mumap;
    metal_kernel kernels[METAL_KERNEL_TYPE_COUNT];
    bool support_simdgroup_reduction;
    bool support_simdgroup_mm;
    bool should_capture_next_compute;
    // dispatch_queue_t d_queue;
};

struct metal_constants {
    float eps; //rms_norm
    float scale; //softmax
    int embed_dim; //embed
};

struct metal_params {
    struct matrix A, B, C, D, bias;
    struct optimization_params opt_params;
    float alpha, beta;
    float16_t half_alpha;
    // batch_size
    int bs;
    // for int4
    float *scales, *offset, *zero_point;
    float16_t *half_scales;
    naive_float16_t *fp16_scales;
    int *int32_zero_point;
    int block_size;
    // for int8 activation
    float *A_scales;
    int8_t A_zero_point;
    // op
    metal_kernel_type op;
    // consts
    float eps; //rms_norm
    float scale; //softmax
    int embed_dim; //embed
    int n_orig_ctx;
    int n_past;
    int n_dims;
    int mode;
    int freq_base;
    int freq_scale;
    int ext_factor;
    int attn_factor;
    int beta_fast;
    int beta_slow;
    //
    int sqlen, past_sqlen, num_heads, head_dim, input_m_dim_z, tgz ;
};

struct metal_cgraph{
    int capacity;
    int n_nodes;
    const struct metal_params ** mm_nodes; // matmul ops (A, B, C)
};

void *allocateSharedMem(size_t size);
void init();
static void metal_free(struct metal_context * ctx);
enum status metal_graph_compute(
    struct metal_cgraph * metal_data);
void add_node(const struct metal_params * new_node);

#endif