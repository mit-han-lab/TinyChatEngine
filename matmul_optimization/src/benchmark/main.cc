#include <math.h>
#include <stdio.h>

#include <cstdlib>
#include <iostream>

#include "lib/matmul.h"

#define BLK_SIZE 16
#define MAX_PRECISION_ERROR 0.01

#define M 1024
#define N 1024
#define K 1024
#define A_ROW M
#define A_COLUMN K
#define B_ROW K
#define B_COLUMN N
#define C_ROW M
#define C_COLUMN N
#define NUM_THREAD 16

float MAT_A[A_ROW * A_COLUMN];
float MAT_B[B_ROW * B_COLUMN];
float transpose_B[B_ROW * B_COLUMN];
float native_C[C_ROW * C_COLUMN];
float output_C[C_ROW * C_COLUMN];

int8_t MAT_A_s8[A_ROW * A_COLUMN];
int8_t MAT_B_s8[B_ROW * B_COLUMN];
int32_t bias_s32[C_COLUMN];
int8_t transpose_B_s8[B_ROW * B_COLUMN];
int8_t native_C_s8[C_ROW * C_COLUMN];
int8_t output_C_s8[C_ROW * C_COLUMN];

bool check_identical(float matA[], float matB[], int size) {
    for (int i = 0; i < size; i++) {
        if (abs((matA[i] - matB[i]) / (matA[i])) > MAX_PRECISION_ERROR) {
            printf("%d: %f, %f", i, matA[i], matB[i]);
            return false;
        }
    }
    return true;
}

bool check_identical(int8_t matA[], int8_t matB[], int size) {
    for (int i = 0; i < size; i++) {
        if (matA[i] != matB[i]) {
            printf("%d: %d, %d", i, matA[i], matB[i]);
            return false;
        }
    }
    return true;
}

template <typename T>
void dump_integer_array(T matA[], int size) {
    for (int i = 0; i < size; i++) {
        printf("%d,", matA[i]);
    }
    printf("\n");
}

void initialize_matrix(float A[], int size) {
    for (int i = 0; i < size; i++) {
        A[i] = (float)(rand()) / (float)(RAND_MAX);
    }
}

void initialize_matrix(int8_t A[], int size) {
    for (int i = 0; i < size; i++) {
        // A[i] = (rand() % 2) - 1;
        A[i] = (rand() % 2);
    }
}

void initialize_matrix(int32_t A[], int size) {
    for (int i = 0; i < size; i++) {
        // A[i] = (rand() % 2) - 1;
        A[i] = (rand() % 2);
    }
}

using namespace matmul;

int main() {
    // initialize
    initialize_matrix(MAT_A, A_ROW * A_COLUMN);
    initialize_matrix(MAT_B, B_ROW * B_COLUMN);
    initialize_matrix(native_C, C_ROW * C_COLUMN);

    initialize_matrix(MAT_A_s8, A_ROW * A_COLUMN);
    initialize_matrix(MAT_B_s8, B_ROW * B_COLUMN);
    initialize_matrix(native_C_s8, C_ROW * C_COLUMN);
    // initialize_matrix(bias_s32, C_ROW * C_COLUMN);

    MatmulOperator matmul_op = MatmulOperator();

    struct matmul_params params, params_int8;
    params.A.row = A_ROW;
    params.A.column = A_COLUMN;
    params.A.data_ptr = MAT_A;
    params.B.row = B_ROW;
    params.B.column = B_COLUMN;
    params.B.data_ptr = MAT_B;
    params.C.row = C_ROW;
    params.C.column = C_COLUMN;
    params.opt_params.blk_size = BLK_SIZE;
    params.opt_params.num_thread = NUM_THREAD;

    // int8
    params_int8.A.row = A_ROW;
    params_int8.A.column = A_COLUMN;
    params_int8.A.int8_data_ptr = MAT_A_s8;
    params_int8.A.qparams.scale = 1.0;
    params_int8.A.qparams.zero_point = 0;
    params_int8.B.row = B_ROW;
    params_int8.B.column = B_COLUMN;
    params_int8.B.int8_data_ptr = MAT_B_s8;
    params_int8.B.qparams.scale = 1.0;
    params_int8.B.qparams.zero_point = 0;
    params_int8.C.row = C_ROW;
    params_int8.C.column = C_COLUMN;
    params_int8.C.int8_data_ptr = native_C_s8;
    params_int8.C.qparams.scale = 1.0;
    params_int8.C.qparams.q_max = 127;
    params_int8.C.qparams.q_min = -128;
    params_int8.C.qparams.zero_point = 0;
    params_int8.opt_params.blk_size = BLK_SIZE;
    params_int8.opt_params.num_thread = NUM_THREAD;
    params_int8.bias.row = 1;
    params_int8.bias.column = C_COLUMN;
    params_int8.bias.int32_data_ptr = bias_s32;

    // Baseline
    params.C.data_ptr = native_C;
    matmul_op.evaluate(MatmulOperator::NAIVE, &params);

    params.C.data_ptr = output_C;
    // unrolling
    matmul_op.evaluate(MatmulOperator::UNROLL, &params);
    if (!check_identical(native_C, output_C, C_ROW * C_COLUMN)) printf("incorrect output of mat_mul_unrolling\n");

    // reordering
    matmul_op.evaluate(MatmulOperator::REORDER, &params);
    if (!check_identical(native_C, output_C, C_ROW * C_COLUMN)) printf("incorrect output of mat_mul_reordering\n");

    // tiling
    matmul_op.evaluate(MatmulOperator::TILING, &params);
    if (!check_identical(native_C, output_C, C_ROW * C_COLUMN)) printf("incorrect output of mat_mul_tiling\n");

    // multithreading
    matmul_op.evaluate(MatmulOperator::MULTITHREAD, &params);
    if (!check_identical(native_C, output_C, C_ROW * C_COLUMN)) printf("incorrect output of mat_mul_multithreading\n");

    // transpose
    matmul_op.evaluate(MatmulOperator::TRANSPOSE, &params);
    if (!check_identical(native_C, output_C, C_ROW * C_COLUMN)) printf("incorrect output of mat_mul_transpose\n");

    // transpose + simd
    initialize_matrix(output_C, C_ROW * C_COLUMN);
    matmul_op.evaluate(MatmulOperator::TRANSPOSE_SIMD, &params);
    if (!check_identical(native_C, output_C, C_ROW * C_COLUMN)) printf("incorrect output of mat_mul_transpose_simd\n");

// cuda
#ifdef CUDA_ENABLE
    matmul_op.evaluate(MatmulOperator::CUDA, &params);
    if (!check_identical(native_C, output_C, C_ROW * C_COLUMN)) printf("incorrect output of mat_mul_cuda\n");
#endif

// ONEDNN
#ifdef ONEDNN_ENABLE
    initialize_matrix(output_C, C_ROW * C_COLUMN);
    matmul_op.evaluate(MatmulOperator::ONEDNN_FP32, &params);
    if (!check_identical(native_C, output_C, C_ROW * C_COLUMN)) printf("\nincorrect output of mat_mul_onedenn\n");
#endif

    // For fast, we need to transpose B first
    for (int i = 0; i < B_COLUMN; i++)
        for (int j = 0; j < B_ROW; j++) transpose_B[i * B_ROW + j] = MAT_B[j * B_COLUMN + i];
    params.B.column = B_ROW;
    params.B.row = B_COLUMN;
    params.B.data_ptr = transpose_B;
    params.opt_params.blk_size = BLK_SIZE;
    params.opt_params.num_thread = NUM_THREAD;

    // fast
    initialize_matrix(output_C, C_ROW * C_COLUMN);
    matmul_op.evaluate(MatmulOperator::FAST, &params);
    if (!check_identical(native_C, output_C, C_ROW * C_COLUMN)) printf("incorrect output of mat_mul_fast\n");

    // int8
    matmul_op.evaluate(MatmulOperator::INT8_BASELINE, &params_int8);

    params_int8.C.int8_data_ptr = output_C_s8;

    // For int8 SIMD, we need to transpose B first
    for (int i = 0; i < B_COLUMN; i++)
        for (int j = 0; j < B_ROW; j++) transpose_B_s8[i * B_ROW + j] = MAT_B_s8[j * B_COLUMN + i];

    params_int8.B.int8_data_ptr = transpose_B_s8;
    initialize_matrix(output_C_s8, C_ROW * C_COLUMN);
    matmul_op.evaluate(MatmulOperator::INT8_AVX, &params_int8);
    if (!check_identical(native_C_s8, output_C_s8, C_ROW * C_COLUMN))
        printf("incorrect output from mat_mul_avx_int8\n");

    initialize_matrix(output_C_s8, C_ROW * C_COLUMN);
    matmul_op.evaluate(MatmulOperator::INT8_AVX_FAST, &params_int8);
    if (!check_identical(native_C_s8, output_C_s8, C_ROW * C_COLUMN))
        printf("incorrect output from mat_mul_avx_int8_fast\n");

    initialize_matrix(output_C_s8, C_ROW * C_COLUMN);
    matmul_op.evaluate(MatmulOperator::INT8_AVX_FAST_2x2, &params_int8);
    if (!check_identical(native_C_s8, output_C_s8, C_ROW * C_COLUMN))
        printf("incorrect output from mat_mul_avx_int8_fast_2x2\n");

    initialize_matrix(output_C_s8, C_ROW * C_COLUMN);
    matmul_op.evaluate(MatmulOperator::INT8_AVX_FAST_2x2_32UNROLL, &params_int8);
    if (!check_identical(native_C_s8, output_C_s8, C_ROW * C_COLUMN))
        printf("incorrect output from mat_mul_avx_int8_fast_2x2_32unroll\n");

    initialize_matrix(output_C_s8, C_ROW * C_COLUMN);
    matmul_op.evaluate(MatmulOperator::INT8_AVX_FAST_2x2_OMP, &params_int8);
    if (!check_identical(native_C_s8, output_C_s8, C_ROW * C_COLUMN))
        printf("incorrect output from mat_mul_avx_int8_fast_2x2_omp\n");

// ONEDNN
#ifdef ONEDNN_ENABLE
    initialize_matrix(output_C_s8, C_ROW * C_COLUMN);
    matmul_op.evaluate(MatmulOperator::ONEDNN_INT8, &params_int8);
    if (!check_identical(native_C_s8, output_C_s8, C_ROW * C_COLUMN))
        printf("incorrect output from mat_mul_onednn_int8\n");
#endif
    // Debugging
    // dump_integer_array(MAT_A_s8, A_ROW * A_COLUMN);
    // dump_integer_array(MAT_B_s8, B_ROW * B_COLUMN);
    // dump_integer_array(native_C_s8, C_ROW * C_COLUMN);
    // dump_integer_array(output_C_s8, C_ROW * C_COLUMN);

    return 0;
}
