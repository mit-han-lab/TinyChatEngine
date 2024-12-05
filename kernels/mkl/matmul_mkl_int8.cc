#include <mkl.h>
#include <mkl_cblas.h>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <iostream>

#include "matmul_mkl.h"
#include "../avx/matmul_avx.h"

namespace matmul{
void mat_mul_mkl_int8_effective_scale(const matmul_params *params) {
    const matrix &A = params->A;
    const matrix &B = params->B;
    matrix &C = const_cast<matrix&>(params->C);

    int M = A.row;
    int K = A.column;
    int N = B.column;

    const int8_t *data_A = A.int8_data_ptr;
    int32_t *data_C = new int32_t[M * N];  // Temporary buffer for accumulation

    const int8_t *data_B = B.int8_data_ptr;

    // Shift A instead of B by adding 128:
    // We shift A because the mkl interface expects A to be unsigned instead of B if row-major
    uint8_t *data_A_shifted = new uint8_t[M * K];
    for (int i = 0; i < M * K; ++i) {
        int16_t temp = static_cast<int16_t>(A.int8_data_ptr[i] + 128) ;
        data_A_shifted[i] = static_cast<uint8_t>(temp);
    }

    MKL_INT lda = K;
    MKL_INT ldb = K;
    MKL_INT ldc = N;


    const MKL_INT8 ao = -(A.qparams.zero_point+128);
    assert(B.qparams.zero_point == 0);
    const MKL_INT8 bo = -(B.qparams.zero_point);  // Adjusted zero point for B
    MKL_INT32 co = C.qparams.zero_point;

    float effective_scale = A.qparams.scale * B.qparams.scale / C.qparams.scale;

    float alpha = 1.0f;
    float beta = 0.0f;
    cblas_gemm_s8u8s32(
        CblasRowMajor, CblasNoTrans, CblasTrans, CblasFixOffset,
        M, N, K,
        alpha,
        data_A_shifted, lda, ao,
        data_B, ldb, bo,
        beta,
        data_C, ldc, &co
    );

    // Post-process the result
    int8_t *data_C_int8 = C.int8_data_ptr;
    for (int i = 0; i < M * N; ++i) {
        int32_t acc = data_C[i];
        acc = static_cast<int32_t>(std::round(acc * effective_scale));
        if (params->bias.int8_data_ptr) {
            acc += static_cast<int32_t>(params->bias.int8_data_ptr[i % N]);
        }

        acc = std::max(static_cast<int32_t>(C.qparams.q_min), acc);
        acc = std::min(static_cast<int32_t>(C.qparams.q_max), acc);

        data_C_int8[i] = static_cast<int8_t>(acc);
    }

    // Clean up
    delete[] data_A_shifted;
    delete[] data_C;
}
void mat_mul_mkl_int8_alpha(const matmul_params *params) {
    const matrix &A = params->A;
    const matrix &B = params->B;
    matrix &C = const_cast<matrix&>(params->C);

    int M = A.row;
    int K = A.column;
    int N = B.column;

    const int8_t *data_A = A.int8_data_ptr;
    int32_t *data_C = new int32_t[M * N];  // Temporary buffer for accumulation

    const int8_t *data_B = B.int8_data_ptr;

    // Shift A instead of B by adding 128:
    // We shift A because the mkl interface expects A to be unsigned instead of B if row-major
    uint8_t *data_A_shifted = new uint8_t[M * K];
    for (int i = 0; i < M * K; ++i) {
        int16_t temp = static_cast<int16_t>(A.int8_data_ptr[i] + 128) ;
        data_A_shifted[i] = static_cast<uint8_t>(temp);
    }

    MKL_INT lda = K;
    MKL_INT ldb = K;
    MKL_INT ldc = N;

    const MKL_INT8 ao = -(A.qparams.zero_point+128);
    assert(B.qparams.zero_point == 0);
    const MKL_INT8 bo = -(B.qparams.zero_point);  // Adjusted zero point for B
    MKL_INT32 co = C.qparams.zero_point;

    float alpha = 1.0f;
    float beta = 0.0f;
    cblas_gemm_s8u8s32(
        CblasRowMajor, CblasNoTrans, CblasTrans, CblasFixOffset,
        M, N, K,
        alpha,
        data_A_shifted, lda, ao,
        data_B, ldb, bo,
        beta,
        data_C, ldc, &co
    );

    // Post-process the result
    int8_t *data_C_int8 = C.int8_data_ptr;
    for (int i = 0; i < M * N; ++i) {
        int32_t acc = data_C[i];
        acc = static_cast<int32_t>(std::round(((float)acc * params->alpha + 
            (float)(params->bias.int8_data_ptr[i % N]) * params->beta)));

        acc = std::max(static_cast<int32_t>(C.qparams.q_min), acc);
        acc = std::min(static_cast<int32_t>(C.qparams.q_max), acc);

        data_C_int8[i] = static_cast<int8_t>(acc);
    }

    // Clean up
    delete[] data_A_shifted;
    delete[] data_C;
}

void MatmulOperatorMKL::mat_mul_accelerator_int8_fast_32unroll_over_column(const struct matmul_params *params) {
    // std::cout << "Running mat_mul_mkl" << std::endl;
    MatmulOperatorAVX fallback;
    fallback.mat_mul_accelerator_int8_fast_32unroll_over_column(params);
    // mat_mul_mkl_int8(params);
}

void MatmulOperatorMKL::mat_mul_accelerator_int8_fast_2x2_32unroll(const struct matmul_params *params) {
    mat_mul_mkl_int8_alpha(params);
    // MatmulOperatorAVX fallback;
    // fallback.mat_mul_accelerator_int8_fast_2x2_32unroll(params);
}

// avx fallback operations
void MatmulOperatorMKL::mat_mul_accelerator_transposed_fastover_column(const struct matmul_params* params) {
    MatmulOperatorAVX fallback;
    fallback.mat_mul_accelerator_transposed_fastover_column(params);
}

void MatmulOperatorMKL::mat_mul_accelerator_transposed_fastover_column_bias(const struct matmul_params* params) {
    MatmulOperatorAVX fallback;
    fallback.mat_mul_accelerator_transposed_fastover_column_bias(params);
}

void MatmulOperatorMKL::mat_mul_accelerator_int8_fast_2x2_32unroll_nobias(const struct matmul_params* params) {
    mat_mul_mkl_int8_effective_scale(params);
}

void MatmulOperatorMKL::mat_mul_accelerator_int8_fast_2x2_32unroll_nobias_batch(const struct matmul_params* params) {
    MatmulOperatorAVX fallback;
    fallback.mat_mul_accelerator_int8_fast_2x2_32unroll_nobias_batch(params);
}

void MatmulOperatorMKL::mat_mul_accelerator_int8_fast_2x2_32unroll_nobias_ofp32(const struct matmul_params* params) {
    MatmulOperatorAVX fallback;
    fallback.mat_mul_accelerator_int8_fast_2x2_32unroll_nobias_ofp32(params);
}

void MatmulOperatorMKL::mat_mul_accelerator_int8_fast_2x2_32unroll_nobias_ofp32_batch(const struct matmul_params* params) {
    MatmulOperatorAVX fallback;
    fallback.mat_mul_accelerator_int8_fast_2x2_32unroll_nobias_ofp32_batch(params);
}

void MatmulOperatorMKL::mat_mul_accelerator_int8_fast_2x2_32unroll_bfp32_ofp32(const struct matmul_params* params) {
    MatmulOperatorAVX fallback;
    fallback.mat_mul_accelerator_int8_fast_2x2_32unroll_bfp32_ofp32(params);
}

void MatmulOperatorMKL::mat_mul_accelerator_int8_fast_2x2_32unroll_bfp32_ofp32_over_column(const struct matmul_params* params) {
    MatmulOperatorAVX fallback;
    fallback.mat_mul_accelerator_int8_fast_2x2_32unroll_bfp32_ofp32_over_column(params);
}

void MatmulOperatorMKL::mat_mul_accelerator_int8_int4_fast_no_offset(struct matmul_params* params) {
    MatmulOperatorAVX fallback;
    fallback.mat_mul_accelerator_int8_int4_fast_no_offset(params);
}

void MatmulOperatorMKL::mat_mul_accelerator_int4_fast(const struct matmul_params* params) {
    MatmulOperatorAVX fallback;
    fallback.mat_mul_accelerator_int4_fast(params);
}

void MatmulOperatorMKL::mat_mul_accelerator_int4_fast_no_offset(const struct matmul_params* params) {
    MatmulOperatorAVX fallback;
    fallback.mat_mul_accelerator_int4_fast_no_offset(params);
}

} // namespace matmul

