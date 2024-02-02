#include <metal_stdlib>
using namespace metal;

using namespace metal;

#define N_SIMDWIDTH 32
#define MAX(x, y) ((x) > (y) ? (x) : (y))
#define MIN(x, y) ((x) < (y) ? (x) : (y))
#define SWAP(x, y) { auto tmp = (x); (x) = (y); (y) = tmp; }

 /* CUDA */
//  __global__ void batch_Add_cuda(Matrix3D<half> input, Matrix3D<half> input2, Matrix3D<half> output) {
//     int i = blockIdx.x * blockDim.x + threadIdx.x;
//     int j = blockIdx.y * blockDim.y + threadIdx.y;
//     int k = blockIdx.z * blockDim.z + threadIdx.z;

//     //// half version
//     if (i < input.m_dim_x && j < input.m_dim_y && k < input.m_dim_z) {
//         output(i, j, k) = __hadd(input(i, j, k), input2(0, j, k));
//     }
// }
kernel void kernel_batch_add(device const float* inputA,
                             device const float* inputB,
                             device float* output,
                             constant MetalMatMulParams& params,
                             uint3 id[[thread_position_in_grid]]) {
    const uint m = param.m_dim_x;
    const uint n = param.m_dim_y;

    const uint idx = id.x;
    const uint idy = id.y;
    const uint idz = id.z;
    output[idx * m * n + idy * n + idz] = inputA[idx * m * n + idy * n + idz] + inputB[idy * n + idz];
}

kernel void kernel_relu(
        device const float * src0,
        device       float * dst,
        uint tpig[[thread_position_in_grid]]) {
    dst[tpig] = max(0.0f, src0[tpig]);
}

 kernel void kernel_silu(
        device const float4 * src0,
        device       float4 * dst,
        uint tpig[[thread_position_in_grid]]) {
    device const float4 & x = src0[tpig];
    dst[tpig] = x / (1.0f + exp(-x));
}

constant float GELU_COEF_A     = 0.044715f;
constant float GELU_QUICK_COEF = -1.702f;
constant float SQRT_2_OVER_PI  = 0.79788456080286535587989211986876f;
kernel void kernel_gelu( 
    device const float4 * src0,
    device       float4 * dst,
    uint tpig[[thread_position_in_grid]]) {
    device const float4 & x = src0[tpig];

    // BEWARE !!!
    // Simply using "tanh" instead of "precise::tanh" will sometimes results in NaNs!
    // This was observed with Falcon 7B and 40B models
    //
    dst[tpig] = 0.5f * x * (1.0f + precise::tanh(SQRT_2_OVER_PI * x * (1.0f + GELU_COEF_A * x * x)));
}
kernel void kernel_gelu_quick(
    device const float4 * src0,
    device       float4 * dst,
    uint tpig[[thread_position_in_grid]]) {
    device const float4 & x = src0[tpig];

    dst[tpig] = x * (1.0f / (1.0f + exp(GELU_QUICK_COEF * x)));
}

// TODO: to be fixed
kernel void kernel_rms_norm(
        device const  void * src0,
        device       float * dst,
        constant MetalMatMulParams& params,
        // constant   int64_t & ne00, // row
        // constant  uint64_t & nb01, // col*sizeof(type)
        // constant     float & eps,
        threadgroup float  * buf [[threadgroup(0)]],
        uint tgpig[[threadgroup_position_in_grid]],
        uint tpitg[[thread_position_in_threadgroup]],
        uint sgitg[[simdgroup_index_in_threadgroup]],
        uint tiisg[[thread_index_in_simdgroup]],
        uint   ntg[[threads_per_threadgroup]]) {
    device const float4 * x = (device const float4 *) ((device const char *) src0 + tgpig*nb01);
    unsigned int ne00 = params.m_dim_x;
    unsigned int nb01 = params.m_dim_y*param.type_size;
    float eps = param.eps;
    float4 sumf = 0;
    float all_sum = 0;

    // parallel sum
    for (int i00 = tpitg; i00 < ne00/4; i00 += ntg) {
        sumf += x[i00] * x[i00]; // take four elements and square it at the same time
    }
    all_sum = sumf[0] + sumf[1] + sumf[2] + sumf[3];
    all_sum = simd_sum(all_sum);
    if (ntg > N_SIMDWIDTH) {
        if (sgitg == 0) {
            buf[tiisg] = 0.0f;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tiisg == 0) {
            buf[sgitg] = all_sum;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        all_sum = buf[tiisg];
        all_sum = simd_sum(all_sum);
    }

    const float mean  = all_sum / ne00;
    const float scale = 1.0f / sqrt(mean + eps);

    device float4 * y = (device float4 *) (dst + tgpig*ne00);
    for (int i00 = tpitg; i00 < ne00/4; i00 += ntg) {
        y[i00] = x[i00] * scale;
    }
}


// TODO: to be fixed
kernel void kernel_soft_max(
        device const float * src0,
        device const float * src1,
        device       float * dst,
        constant MetalMatMulParams& params,
        // constant   int64_t & ne00,
        // constant   int64_t & ne01,
        // constant   int64_t & ne02,
        // constant     float & scale,
        threadgroup float  * buf [[threadgroup(0)]],
        uint  tgpig[[threadgroup_position_in_grid]],
        uint  tpitg[[thread_position_in_threadgroup]],
        uint  sgitg[[simdgroup_index_in_threadgroup]],
        uint  tiisg[[thread_index_in_simdgroup]],
        uint    ntg[[threads_per_threadgroup]]) {
    const int64_t ne00 = params.m_dim_x;
    const int64_t ne01 = params.m_dim_y;
    const int64_t ne02 = params.m_dim_z;
    const int64_t scale = params.scale;


    const int64_t i03 = (tgpig) / (ne02*ne01);
    const int64_t i02 = (tgpig - i03*ne02*ne01) / ne01;
    const int64_t i01 = (tgpig - i03*ne02*ne01 - i02*ne01);

    device const float * psrc0 =         src0 + i03*ne02*ne01*ne00 + i02*ne01*ne00 + i01*ne00;
    device const float * pmask = src1 != src0 ? src1                               + i01*ne00 : nullptr;
    device       float * pdst  =         dst  + i03*ne02*ne01*ne00 + i02*ne01*ne00 + i01*ne00;

    // parallel max
    float lmax = -INFINITY;

    for (int i00 = tpitg; i00 < ne00; i00 += ntg) {
        lmax = MAX(lmax, psrc0[i00]*scale + (pmask ? pmask[i00] : 0.0f));
    }

    // find the max value in the block
    float max_val = simd_max(lmax);
    if (ntg > N_SIMDWIDTH) {
        if (sgitg == 0) {
            buf[tiisg] = -INFINITY;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tiisg == 0) {
            buf[sgitg] = max_val;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        max_val = buf[tiisg];
        max_val = simd_max(max_val);
    }

    // parallel sum
    float lsum = 0.0f;
    for (int i00 = tpitg; i00 < ne00; i00 += ntg) {
        const float exp_psrc0 = exp((psrc0[i00]*scale + (pmask ? pmask[i00] : 0.0f)) - max_val);
        lsum += exp_psrc0;
        pdst[i00] = exp_psrc0;
    }

    // This barrier fixes a failing test
    // ref: https://github.com/ggerganov/ggml/pull/621#discussion_r1425156335
    threadgroup_barrier(mem_flags::mem_none);

    float sum = simd_sum(lsum);

    if (ntg > N_SIMDWIDTH) {
        if (sgitg == 0) {
            buf[tiisg] = 0.0f;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tiisg == 0) {
            buf[sgitg] = sum;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        sum = buf[tiisg];
        sum = simd_sum(sum);
    }

    const float inv_sum = 1.0f/sum;

    for (int i00 = tpitg; i00 < ne00; i00 += ntg) {
        pdst[i00] *= inv_sum;
    }
}

// TODO: to be fixed
kernel void kernel_soft_max_4(
        device const float * src0,
        device const float * src1,
        device       float * dst,
        constant MetalMatMulParams& params,
        // constant   int64_t & ne00,
        // constant   int64_t & ne01,
        // constant   int64_t & ne02,
        // constant     float & scale,
        threadgroup float  * buf [[threadgroup(0)]],
        uint  tgpig[[threadgroup_position_in_grid]],
        uint  tpitg[[thread_position_in_threadgroup]],
        uint  sgitg[[simdgroup_index_in_threadgroup]],
        uint  tiisg[[thread_index_in_simdgroup]],
        uint    ntg[[threads_per_threadgroup]]) {
    const int64_t ne00 = params.m_dim_x;
    const int64_t ne01 = params.m_dim_y;
    const int64_t ne02 = params.m_dim_z;
    const int64_t scale = params.scale;

    const int64_t i03 = (tgpig) / (ne02*ne01);
    const int64_t i02 = (tgpig - i03*ne02*ne01) / ne01;
    const int64_t i01 = (tgpig - i03*ne02*ne01 - i02*ne01);

    device const float4 * psrc4 =                (device const float4 *)(src0 + i03*ne02*ne01*ne00 + i02*ne01*ne00 + i01*ne00);
    device const float4 * pmask = src1 != src0 ? (device const float4 *)(src1 +                                      i01*ne00) : nullptr;
    device       float4 * pdst4 =                (device       float4 *)(dst  + i03*ne02*ne01*ne00 + i02*ne01*ne00 + i01*ne00);

    // parallel max
    float4 lmax4 = -INFINITY;

    for (int i00 = tpitg; i00 < ne00/4; i00 += ntg) {
        lmax4 = fmax(lmax4, psrc4[i00]*scale + (pmask ? pmask[i00] : 0.0f));
    }

    const float lmax = MAX(MAX(lmax4[0], lmax4[1]), MAX(lmax4[2], lmax4[3]));

    float max_val = simd_max(lmax);
    if (ntg > N_SIMDWIDTH) {
        if (sgitg == 0) {
            buf[tiisg] = -INFINITY;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tiisg == 0) {
            buf[sgitg] = max_val;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        max_val = buf[tiisg];
        max_val = simd_max(max_val);
    }

    // parallel sum
    float4 lsum4 = 0.0f;
    for (int i00 = tpitg; i00 < ne00/4; i00 += ntg) {
        const float4 exp_psrc4 = exp((psrc4[i00]*scale + (pmask ? pmask[i00] : 0.0f)) - max_val);
        lsum4 += exp_psrc4;
        pdst4[i00] = exp_psrc4;
    }

    const float lsum = lsum4[0] + lsum4[1] + lsum4[2] + lsum4[3];

    // This barrier fixes a failing test
    // ref: https://github.com/ggerganov/ggml/pull/621#discussion_r1425156335
    threadgroup_barrier(mem_flags::mem_none);

    float sum = simd_sum(lsum);

    if (ntg > N_SIMDWIDTH) {
        if (sgitg == 0) {
            buf[tiisg] = 0.0f;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tiisg == 0) {
            buf[sgitg] = sum;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        sum = buf[tiisg];
        sum = simd_sum(sum);
    }

    const float inv_sum = 1.0f/sum;

    for (int i00 = tpitg; i00 < ne00/4; i00 += ntg) {
        pdst4[i00] *= inv_sum;
    }
}


// ROPE //
static float rope_yarn_ramp(const float low, const float high, const int i0) {
    const float y = (i0 / 2 - low) / max(0.001f, high - low);
    return 1.0f - min(1.0f, max(0.0f, y));
}

// YaRN algorithm based on LlamaYaRNScaledRotaryEmbedding.py from https://github.com/jquesnelle/yarn
// MIT licensed. Copyright (c) 2023 Jeffrey Quesnelle and Bowen Peng.
static void rope_yarn( float theta_extrap, float freq_scale, float corr_dims[2], int64_t i0, float ext_factor, float mscale,
    thread float * cos_theta, thread float * sin_theta) {
    // Get n-d rotational scaling corrected for extrapolation
    float theta_interp = freq_scale * theta_extrap;
    float theta = theta_interp;
    if (ext_factor != 0.0f) {
        float ramp_mix = rope_yarn_ramp(corr_dims[0], corr_dims[1], i0) * ext_factor;
        theta = theta_interp * (1 - ramp_mix) + theta_extrap * ramp_mix;

        // Get n-d magnitude scaling corrected for interpolation
        mscale *= 1.0f + 0.1f * log(1.0f / freq_scale);
    }
    *cos_theta = cos(theta) * mscale;
    *sin_theta = sin(theta) * mscale;
}

// Apparently solving `n_rot = 2pi * x * base^((2 * max_pos_emb) / n_dims)` for x, we get
// `corr_fac(n_rot) = n_dims * log(max_pos_emb / (n_rot * 2pi)) / (2 * log(base))`
static float rope_yarn_corr_factor(int n_dims, int n_orig_ctx, float n_rot, float base) {
    return n_dims * log(n_orig_ctx / (n_rot * 2 * M_PI_F)) / (2 * log(base));
}

static void rope_yarn_corr_dims(
    int n_dims, int n_orig_ctx, float freq_base, float beta_fast, float beta_slow, float dims[2]) {
    // start and end correction dims
    dims[0] = max(0.0f,         floor(rope_yarn_corr_factor(n_dims, n_orig_ctx, beta_fast, freq_base)));
    dims[1] = min(n_dims - 1.0f, ceil(rope_yarn_corr_factor(n_dims, n_orig_ctx, beta_slow, freq_base)));
}

// typedef void (rope_t)(
//         device const    void * src0,
//         device const int32_t * src1,
//         device         float * dst,
//         constant     int64_t & ne00,
//         constant     int64_t & ne01,
//         constant     int64_t & ne02,
//         constant     int64_t & ne03,
//         constant    uint64_t & nb00,
//         constant    uint64_t & nb01,
//         constant    uint64_t & nb02,
//         constant    uint64_t & nb03,
//         constant     int64_t & ne0,
//         constant     int64_t & ne1,
//         constant     int64_t & ne2,
//         constant     int64_t & ne3,
//         constant    uint64_t & nb0,
//         constant    uint64_t & nb1,
//         constant    uint64_t & nb2,
//         constant    uint64_t & nb3,
//         constant         int & n_past,
//         constant         int & n_dims,
//         constant         int & mode,
//         constant         int & n_orig_ctx,
//         constant       float & freq_base,
//         constant       float & freq_scale,
//         constant       float & ext_factor,
//         constant       float & attn_factor,
//         constant       float & beta_fast,
//         constant       float & beta_slow,
//         uint  tiitg[[thread_index_in_threadgroup]],
//         uint3 tptg[[threads_per_threadgroup]],
//         uint3 tgpig[[threadgroup_position_in_grid]]);

// // TODO: to be fixed
// template<typename T>
kernel void kernel_rope(
        device const    void * src0,
        device const int32_t * src1,
        device         float * dst,
        constant MetalMatMulParams& params,
        // constant     int64_t & ne00,
        // constant     int64_t & ne01,
        // constant     int64_t & ne02,
        // constant     int64_t & ne03,
        // constant    uint64_t & nb00,
        // constant    uint64_t & nb01,
        // constant    uint64_t & nb02,
        // constant    uint64_t & nb03,
        // constant     int64_t & ne0,
        // constant     int64_t & ne1,
        // constant     int64_t & ne2,
        // constant     int64_t & ne3,
        // constant    uint64_t & nb0,
        // constant    uint64_t & nb1,
        // constant    uint64_t & nb2,
        // constant    uint64_t & nb3,
        // constant         int & n_past,
        // constant         int & n_dims,
        // constant         int & mode,
        // constant         int & n_orig_ctx,
        // constant       float & freq_base,
        // constant       float & freq_scale,
        // constant       float & ext_factor,
        // constant       float & attn_factor,
        // constant       float & beta_fast,
        // constant       float & beta_slow,
        uint  tiitg[[thread_index_in_threadgroup]],
        uint3 tptg[[threads_per_threadgroup]],
        uint3 tgpig[[threadgroup_position_in_grid]]) {
    constant     int64_t ne00 = param.m_dim_x;
    constant     int64_t ne01 = param.m_dim_y;
    constant     int64_t ne02 = param.m_dim_z;
    constant     int64_t ne03 = 0;
    constant    uint64_t nb00 = param.m_dim_x*param.type_size;
    constant    uint64_t nb01 = param.m_dim_y*param.type_size;
    constant    uint64_t nb02 = param.m_dim_z*param.type_size;
    constant    uint64_t nb03 = 0;
    constant     int64_t ne0 = param.m_dim_x;
    constant     int64_t ne1 = param.m_dim_y;
    constant     int64_t ne2 = param.m_dim_z;
    constant     int64_t ne3 = 0;
    constant    uint64_t nb0 = param.m_dim_x*param.type_size;
    constant    uint64_t nb1 = param.m_dim_y*param.type_size;
    constant    uint64_t nb2 = param.m_dim_z*param.type_size;
    constant    uint64_t nb3 = 0;

    int n_past = param.n_past;
    int n_dims = param.n_dims;
    int mode = param.mode;
    int n_orig_ctx = param.n_orig_ctx;
    float freq_base = param.freq_base;
    float freq_scale = param.freq_scale;
    float ext_factor = param.ext_factor;
    float attn_factor = param.attn_factor;
    float beta_fast = param.beta_fast;
    float beta_slow = param.beta_slow;


    const int64_t i3 = tgpig[2];
    const int64_t i2 = tgpig[1];
    const int64_t i1 = tgpig[0];

    const bool is_neox = mode & 2;

    float corr_dims[2];
    rope_yarn_corr_dims(n_dims, n_orig_ctx, freq_base, beta_fast, beta_slow, corr_dims);

    device const int32_t * pos = src1;

    const int64_t p = pos[i2];

    const float theta_0 = (float)p;
    const float inv_ndims = -1.f/n_dims;

    if (!is_neox) {
        for (int64_t i0 = 2*tiitg; i0 < ne0; i0 += 2*tptg.x) {

            const float theta = theta_0 * pow(freq_base, inv_ndims*i0);
            float cos_theta, sin_theta;
            rope_yarn(theta, freq_scale, corr_dims, i0, ext_factor, attn_factor, &cos_theta, &sin_theta);

            device const T * const src = (device T *)((device char *) src0 + i3*nb03 + i2*nb02 + i1*nb01 + i0*nb00);
            device       T * dst_data  = (device T *)((device char *)  dst + i3*nb3  + i2*nb2  + i1*nb1  + i0*nb0);

            const T x0 = src[0];
            const T x1 = src[1];

            dst_data[0] = x0*cos_theta - x1*sin_theta;
            dst_data[1] = x0*sin_theta + x1*cos_theta;
        }
    } else {
        for (int64_t ic = 2*tiitg; ic < ne0; ic += 2*tptg.x) {
            if (ic < n_dims) {
                const int64_t ib = 0;

                // simplified from `(ib * n_dims + ic) * inv_ndims`
                const float cur_rot = inv_ndims*ic - ib;

                const float theta = theta_0 * pow(freq_base, cur_rot);
                float cos_theta, sin_theta;
                rope_yarn(theta, freq_scale, corr_dims, cur_rot, ext_factor, attn_factor, &cos_theta, &sin_theta);

                const int64_t i0 = ib*n_dims + ic/2;

                device const T * const src = (device T *)((device char *) src0 + i3*nb03 + i2*nb02 + i1*nb01 + i0*nb00);
                device       T * dst_data  = (device T *)((device char *)  dst + i3*nb3  + i2*nb2  + i1*nb1  + i0*nb0);

                const float x0 = src[0];
                const float x1 = src[n_dims/2];

                dst_data[0]        = x0*cos_theta - x1*sin_theta;
                dst_data[n_dims/2] = x0*sin_theta + x1*cos_theta;
            } else {
                const int64_t i0 = ic;

                device const T * const src = (device T *)((device char *) src0 + i3*nb03 + i2*nb02 + i1*nb01 + i0*nb00);
                device       T * dst_data  = (device T *)((device char *)  dst + i3*nb3  + i2*nb2  + i1*nb1  + i0*nb0);

                dst_data[0] = src[0];
                dst_data[1] = src[1];
            }
        }
    }
}

// template [[host_name("kernel_rope_f32")]] kernel rope_t kernel_rope<float>;
// template [[host_name("kernel_rope_f16")]] kernel rope_t kernel_rope<half>;

/*
Performance comparision with the test case:
CPU: 4000ms, ~60GOP/s
matmulInt4: 8000ms, ~30GOP/s
matmulInt4_SIMD_Q4Interleave(GPU): 3000 ms, ~83 GOP/s
matmulInt4_SIMD_Q4Interleave_unroll16(GPU): 1800 ms, 133 GOP/s
matmulInt4_SIMD_Q4Interleave_unroll32(GPU): 1500 ms, 160 GOP/s
*/

#include "opParams.h"
kernel void matmul(device const float* inA,
                    device const float* inB, // column major
                    device float* result,
                    constant MetalMatMulParams& params,
                    uint2 id [[thread_position_in_grid]])
{
    // the for-loop is replaced with a collection of threads, each of which
    // calls this function.

    const uint n = params.n;
    const uint k = params.k;

    const uint idx = id.x; // column index of the output
    const uint idy = id.y; // row index of the output

    float sum = 0;
    for (uint i = 0; i < k; i++){
        float vA = inA[idy * k + i];
        float vB = inB[idx * k + i];

        sum += vA * vB;
    }
    result[idy * n + idx] = sum;
}

kernel void matmulInt4(device const float* inA,
                    device const uint8_t* inB, // column major
                    device float* result,
                    device const float* scales,
                    constant MetalMatMulParams& params,
                    uint2 id [[thread_position_in_grid]])
{
    // the for-loop is replaced with a collection of threads, each of which
    // calls this function.

    const uint n = params.n;
    const uint k = params.k;
    const uint group_size = params.group_size;

    const uint idx = id.x; // column index of the output
    const uint idy = id.y; // row index of the output

    float sum = 0;
    for (uint i = 0; i < k; i += group_size){
        float scale = scales[(idx * k + i) / group_size];
        for (uint j = 0; j < group_size; j+=2){
            size_t weight_idx = (idx * k + i + j) / 2;
            uint8_t weight_packed = inB[weight_idx];
            int8_t vl = (weight_packed & 0x0F) - 8;
            int8_t vh = (weight_packed >> 4) - 8;

            sum += (inA[idy * k + i + j] * vl) * scale;
            sum += (inA[idy * k + i + j + 1] * vh) * scale;
        }
    }
    result[idy * n + idx] = sum;
}


kernel void matmulInt4_SIMD_Q4Interleave(
                    device const packed_float4* inA,
                    device const packed_char4* inB, // column major
                    device float* result,
                    device const float* scales,
                    constant MetalMatMulParams& params,
                    uint2 id [[thread_position_in_grid]])
{
    // the for-loop is replaced with a collection of threads, each of which
    // calls this function.

    const uint n = params.n;
    const uint k = params.k;
    const uint group_size = params.group_size;

    const uint idx = id.x; // column index of the output
    const uint idy = id.y; // row index of the output

    packed_char4 lowMask = {0x0F, 0x0F, 0x0F, 0x0F};
    packed_float4 sum4 = {0.0f, 0.0f, 0.0f, 0.0f};

    for (uint i = 0; i < k; i += group_size){
        float scale = scales[(idx * k + i) / group_size];
        packed_float4 scale4 = {scale, scale, scale, scale};
        for (uint j = 0; j < group_size; j+= 8){
            // sequential: (a, b), (c, d), (e, f), (g, h): 32 bit = 4xuint8
            // expected layout of inB: (a, e), (b, f), (c, g), (d, h)
            // low; (a, 0), (b, 0), (c, 0), (d, 0)
            // high: (e, 0), (f, 0), (g, 0), (h, 0)
            size_t weight_idx = (idx * k + i + j) / 8;
            size_t activation_idx = (idy * k + i + j) / 4;
            packed_char4 packed_8 = inB[weight_idx];
            packed_char4 packed_low = packed_8 & lowMask;
            packed_char4 packed_high = (packed_8 >> 4) & lowMask;

            packed_float4 inAlow = inA[activation_idx];
            packed_float4 inAhigh = inA[activation_idx+1];
            packed_float4 inBlow = packed_float4(packed_low) * scale4;
            packed_float4 inBhigh = packed_float4(packed_high) * scale4;

            sum4 += inAlow * inBlow;
            sum4 += inAhigh * inBhigh;
        }
    }
    float sum = sum4[0] + sum4[1] + sum4[2] + sum4[3];
    result[idy * n + idx] = sum;
}

kernel void matmulUInt4_SIMD_Q4Interleave_unroll16(
                    device const packed_float4* inA,
                    device const packed_char4* inB, // column major
                    device float* result,
                    device const float* scales,
                    constant MetalMatMulParams& params,
                    uint2 id [[thread_position_in_grid]])
{
    // the for-loop is replaced with a collection of threads, each of which
    // calls this function.

    const uint n = params.n;
    const uint k = params.k;
    const uint group_size = params.group_size;

    const uint idx = id.x; // column index of the output
    const uint idy = id.y; // row index of the output

    packed_char4 lowMask = {0x0F, 0x0F, 0x0F, 0x0F};
    packed_float4 sum4 = {0.0f, 0.0f, 0.0f, 0.0f};
    packed_char4 offsets = {8, 8, 8, 8};

    for (uint i = 0; i < k; i += group_size){
        float scale = scales[(idx * k + i) / group_size];
        packed_float4 scale4 = {scale, scale, scale, scale};
        for (uint j = 0; j < group_size; j+= 16){
            // sequential: (a, b), (c, d), (e, f), (g, h): 32 bit = 4xuint8
            // expected layout of inB: (a, e), (b, f), (c, g), (d, h)
            // low; (a, 0), (b, 0), (c, 0), (d, 0)
            // high: (e, 0), (f, 0), (g, 0), (h, 0)
            size_t weight_idx = (idx * k + i + j) / 8;
            size_t activation_idx = (idy * k + i + j) / 4;
            packed_char4 packed_8_0 = inB[weight_idx];
            packed_char4 packed_8_1 = inB[weight_idx + 1];
            packed_char4 packed_low_0 = (packed_8_0 & lowMask) - offsets;;
            packed_char4 packed_low_1 = (packed_8_1 & lowMask) - offsets;;
            packed_char4 packed_high_0 = ((packed_8_0 >> 4) & lowMask) - offsets;
            packed_char4 packed_high_1 = ((packed_8_1 >> 4) & lowMask) - offsets;

            packed_float4 inAlow_0 = inA[activation_idx];
            packed_float4 inAlow_1 = inA[activation_idx+2];
            packed_float4 inAhigh_0 = inA[activation_idx+1];
            packed_float4 inAhigh_1 = inA[activation_idx+3];
            packed_float4 inBlow_0 = packed_float4(packed_low_0) * scale4;
            packed_float4 inBlow_1 = packed_float4(packed_low_1) * scale4;
            packed_float4 inBhigh_0 = packed_float4(packed_high_0) * scale4;
            packed_float4 inBhigh_1 = packed_float4(packed_high_1) * scale4;

            sum4 += inAlow_0 * inBlow_0;
            sum4 += inAlow_1 * inBlow_1;
            sum4 += inAhigh_0 * inBhigh_0;
            sum4 += inAhigh_1 * inBhigh_1;
        }
    }
    float sum = sum4[0] + sum4[1] + sum4[2] + sum4[3];
    result[idy * n + idx] = sum;
}


kernel void matmulUInt4_SIMD_Q4Interleave_unroll32(
                    device const packed_float4* inA,
                    device const packed_char4* inB, // column major
                    device float* result,
                    device const float* scales,
                    constant MetalMatMulParams& params,
                    uint2 id [[thread_position_in_grid]])
{
    // the for-loop is replaced with a collection of threads, each of which
    // calls this function.

    const uint n = params.n;
    const uint k = params.k;
    const uint group_size = params.group_size;

    const uint idx = id.x; // column index of the output
    const uint idy = id.y; // row index of the output

    packed_char4 lowMask = {0x0F, 0x0F, 0x0F, 0x0F};
    packed_char4 offsets = {8, 8, 8, 8};
    packed_float4 sum4 = {0.0f, 0.0f, 0.0f, 0.0f};

    for (uint i = 0; i < k; i += group_size){
        float scale = scales[(idx * k + i) / group_size];
        packed_float4 scale4 = {scale, scale, scale, scale};
        for (uint j = 0; j < group_size; j+= 32){
            // sequential: (a, b), (c, d), (e, f), (g, h): 32 bit = 4xuint8
            // expected layout of inB: (a, e), (b, f), (c, g), (d, h)
            // low; (a, 0), (b, 0), (c, 0), (d, 0)
            // high: (e, 0), (f, 0), (g, 0), (h, 0)
            size_t weight_idx = (idx * k + i + j) / 8;
            size_t activation_idx = (idy * k + i + j) / 4;
            packed_char4 packed_8_0 = inB[weight_idx];
            packed_char4 packed_8_1 = inB[weight_idx + 1];
            packed_char4 packed_8_2 = inB[weight_idx + 2];
            packed_char4 packed_8_3 = inB[weight_idx + 3];

            packed_char4 packed_low_0 = (packed_8_0 & lowMask) - offsets;;
            packed_char4 packed_low_1 = (packed_8_1 & lowMask) - offsets;;
            packed_char4 packed_low_2 = (packed_8_2 & lowMask) - offsets;;
            packed_char4 packed_low_3 = (packed_8_3 & lowMask) - offsets;;

            packed_char4 packed_high_0 = ((packed_8_0 >> 4) & lowMask) - offsets;
            packed_char4 packed_high_1 = ((packed_8_1 >> 4) & lowMask) - offsets;
            packed_char4 packed_high_2 = ((packed_8_2 >> 4) & lowMask) - offsets;
            packed_char4 packed_high_3 = ((packed_8_3 >> 4) & lowMask) - offsets;

            packed_float4 inAlow_0 = inA[activation_idx];
            packed_float4 inAhigh_0 = inA[activation_idx+1];
            packed_float4 inAlow_1 = inA[activation_idx+2];
            packed_float4 inAhigh_1 = inA[activation_idx+3];
            packed_float4 inAlow_2 = inA[activation_idx+4];
            packed_float4 inAhigh_2 = inA[activation_idx+5];
            packed_float4 inAlow_3 = inA[activation_idx+6];
            packed_float4 inAhigh_3 = inA[activation_idx+7];

            packed_float4 inBlow_0 = packed_float4(packed_low_0) * scale4;
            packed_float4 inBlow_1 = packed_float4(packed_low_1) * scale4;
            packed_float4 inBlow_2 = packed_float4(packed_low_2) * scale4;
            packed_float4 inBlow_3 = packed_float4(packed_low_3) * scale4;

            packed_float4 inBhigh_0 = packed_float4(packed_high_0) * scale4;
            packed_float4 inBhigh_1 = packed_float4(packed_high_1) * scale4;
            packed_float4 inBhigh_2 = packed_float4(packed_high_2) * scale4;
            packed_float4 inBhigh_3 = packed_float4(packed_high_3) * scale4;

            sum4 += inAlow_0 * inBlow_0;
            sum4 += inAlow_1 * inBlow_1;
            sum4 += inAlow_2 * inBlow_2;
            sum4 += inAlow_3 * inBlow_3;
            sum4 += inAhigh_0 * inBhigh_0;
            sum4 += inAhigh_1 * inBhigh_1;
            sum4 += inAhigh_2 * inBhigh_2;
            sum4 += inAhigh_3 * inBhigh_3;
        }
    }
    float sum = sum4[0] + sum4[1] + sum4[2] + sum4[3];
    result[idy * n + idx] = sum;
}

kernel void matmulUInt4_SIMD_Q4Interleave_unroll2x32(
                    device const packed_float4* inA,
                    device const packed_char4* inB, // column major
                    device float* result,
                    device const float* scales,
                    constant MetalMatMulParams& params,
                    uint2 id [[thread_position_in_grid]])
{
    // the for-loop is replaced with a collection of threads, each of which
    // calls this function.

    const uint n = params.n;
    const uint k = params.k;
    const uint group_size = params.group_size;

    const uint idx = id.x; // column index of the output
    const uint idy = id.y; // row index of the output

    packed_char4 lowMask = {0x0F, 0x0F, 0x0F, 0x0F};
    packed_char4 offsets = {8, 8, 8, 8};
    packed_float4 sum4 = {0.0f, 0.0f, 0.0f, 0.0f};
    packed_float4 sum4_col2 = {0.0f, 0.0f, 0.0f, 0.0f};

    packed_float4 a;

    for (uint i = 0; i < k; i += group_size){
        float scale = scales[(idx * k + i) / group_size];
        float scale_col2 = scales[((idx+1) * k + i) / group_size];
        packed_float4 scale4 = {scale, scale, scale, scale};
        packed_float4 scale4_col2 = {scale_col2, scale_col2, scale_col2, scale_col2};
        for (uint j = 0; j < group_size; j+= 32){
            // sequential: (a, b), (c, d), (e, f), (g, h): 32 bit = 4xuint8
            // expected layout of inB: (a, e), (b, f), (c, g), (d, h)
            // low; (a, 0), (b, 0), (c, 0), (d, 0)
            // high: (e, 0), (f, 0), (g, 0), (h, 0)
            size_t weight_idx = (idx * k + i + j) / 8;
            size_t weight_col2_idx = ((idx+1) * k + i + j) / 8;
            size_t activation_idx = (idy * k + i + j) / 4;
            packed_char4 packed_8_0 = inB[weight_idx];
            packed_char4 packed_8_1 = inB[weight_idx + 1];
            packed_char4 packed_8_2 = inB[weight_idx + 2];
            packed_char4 packed_8_3 = inB[weight_idx + 3];

            packed_char4 packed_low_0 = (packed_8_0 & lowMask) - offsets;
            packed_char4 packed_low_1 = (packed_8_1 & lowMask) - offsets;
            packed_char4 packed_low_2 = (packed_8_2 & lowMask) - offsets;
            packed_char4 packed_low_3 = (packed_8_3 & lowMask) - offsets;

            packed_char4 packed_high_0 = ((packed_8_0 >> 4) & lowMask) - offsets;
            packed_char4 packed_high_1 = ((packed_8_1 >> 4) & lowMask) - offsets;
            packed_char4 packed_high_2 = ((packed_8_2 >> 4) & lowMask) - offsets;
            packed_char4 packed_high_3 = ((packed_8_3 >> 4) & lowMask) - offsets;

            packed_float4 inAlow_0 = inA[activation_idx];
            packed_float4 inAhigh_0 = inA[activation_idx+1];
            packed_float4 inAlow_1 = inA[activation_idx+2];
            packed_float4 inAhigh_1 = inA[activation_idx+3];
            packed_float4 inAlow_2 = inA[activation_idx+4];
            packed_float4 inAhigh_2 = inA[activation_idx+5];
            packed_float4 inAlow_3 = inA[activation_idx+6];
            packed_float4 inAhigh_3 = inA[activation_idx+7];

            packed_float4 inBlow_0 = packed_float4(packed_low_0) * scale4;
            packed_float4 inBlow_1 = packed_float4(packed_low_1) * scale4;
            packed_float4 inBlow_2 = packed_float4(packed_low_2) * scale4;
            packed_float4 inBlow_3 = packed_float4(packed_low_3) * scale4;

            packed_float4 inBhigh_0 = packed_float4(packed_high_0) * scale4;
            packed_float4 inBhigh_1 = packed_float4(packed_high_1) * scale4;
            packed_float4 inBhigh_2 = packed_float4(packed_high_2) * scale4;
            packed_float4 inBhigh_3 = packed_float4(packed_high_3) * scale4;

            sum4 += inAlow_0 * inBlow_0;
            sum4 += inAlow_1 * inBlow_1;
            sum4 += inAlow_2 * inBlow_2;
            sum4 += inAlow_3 * inBlow_3;
            sum4 += inAhigh_0 * inBhigh_0;
            sum4 += inAhigh_1 * inBhigh_1;
            sum4 += inAhigh_2 * inBhigh_2;
            sum4 += inAhigh_3 * inBhigh_3;

        }
    }
    float sum = sum4[0] + sum4[1] + sum4[2] + sum4[3];
    result[idy * n + idx] = sum;
}

kernel void matmulUInt4_SIMD_Q4Interleave_half_unroll32(
                    device const packed_half4* inA,
                    device const packed_char4* inB, // column major
                    device float* result,
                    device const float* scales,
                    constant MetalMatMulParams& params,
                    uint2 id [[thread_position_in_grid]])
{
    // the for-loop is replaced with a collection of threads, each of which
    // calls this function.

    const uint n = params.n;
    const uint k = params.k;
    const uint group_size = params.group_size;

    const uint idx = id.x; // column index of the output
    const uint idy = id.y; // row index of the output

    packed_char4 lowMask = {0x0F, 0x0F, 0x0F, 0x0F};
    packed_char4 offsets = {8, 8, 8, 8};
    packed_half4 sum4 = {0.0f, 0.0f, 0.0f, 0.0f};

    for (uint i = 0; i < k; i += group_size){
        half scale = half(scales[(idx * k + i) / group_size]);
        packed_half4 scale4 = {scale, scale, scale, scale};
        for (uint j = 0; j < group_size; j+= 32){
            // sequential: (a, b), (c, d), (e, f), (g, h): 32 bit = 4xuint8
            // expected layout of inB: (a, e), (b, f), (c, g), (d, h)
            // low; (a, 0), (b, 0), (c, 0), (d, 0)
            // high: (e, 0), (f, 0), (g, 0), (h, 0)
            size_t weight_idx = (idx * k + i + j) / 8;
            size_t activation_idx = (idy * k + i + j) / 4;
            packed_char4 packed_8_0 = inB[weight_idx];
            packed_char4 packed_8_1 = inB[weight_idx + 1];
            packed_char4 packed_8_2 = inB[weight_idx + 2];
            packed_char4 packed_8_3 = inB[weight_idx + 3];

            packed_char4 packed_low_0 = (packed_8_0 & lowMask) - offsets;;
            packed_char4 packed_low_1 = (packed_8_1 & lowMask) - offsets;;
            packed_char4 packed_low_2 = (packed_8_2 & lowMask) - offsets;;
            packed_char4 packed_low_3 = (packed_8_3 & lowMask) - offsets;;

            packed_char4 packed_high_0 = ((packed_8_0 >> 4) & lowMask) - offsets;
            packed_char4 packed_high_1 = ((packed_8_1 >> 4) & lowMask) - offsets;
            packed_char4 packed_high_2 = ((packed_8_2 >> 4) & lowMask) - offsets;
            packed_char4 packed_high_3 = ((packed_8_3 >> 4) & lowMask) - offsets;

            packed_half4 inAlow_0 = inA[activation_idx];
            packed_half4 inAhigh_0 = inA[activation_idx+1];
            packed_half4 inAlow_1 = inA[activation_idx+2];
            packed_half4 inAhigh_1 = inA[activation_idx+3];
            packed_half4 inAlow_2 = inA[activation_idx+4];
            packed_half4 inAhigh_2 = inA[activation_idx+5];
            packed_half4 inAlow_3 = inA[activation_idx+6];
            packed_half4 inAhigh_3 = inA[activation_idx+7];

            packed_half4 inBlow_0 = packed_half4(packed_low_0) * scale4;
            packed_half4 inBlow_1 = packed_half4(packed_low_1) * scale4;
            packed_half4 inBlow_2 = packed_half4(packed_low_2) * scale4;
            packed_half4 inBlow_3 = packed_half4(packed_low_3) * scale4;

            packed_half4 inBhigh_0 = packed_half4(packed_high_0) * scale4;
            packed_half4 inBhigh_1 = packed_half4(packed_high_1) * scale4;
            packed_half4 inBhigh_2 = packed_half4(packed_high_2) * scale4;
            packed_half4 inBhigh_3 = packed_half4(packed_high_3) * scale4;

            sum4 += inAlow_0 * inBlow_0;
            sum4 += inAlow_1 * inBlow_1;
            sum4 += inAlow_2 * inBlow_2;
            sum4 += inAlow_3 * inBlow_3;
            sum4 += inAhigh_0 * inBhigh_0;
            sum4 += inAhigh_1 * inBhigh_1;
            sum4 += inAhigh_2 * inBhigh_2;
            sum4 += inAhigh_3 * inBhigh_3;
        }
    }
    half sum = sum4[0] + sum4[1] + sum4[2] + sum4[3];
    result[idy * n + idx] = float(sum);
}
