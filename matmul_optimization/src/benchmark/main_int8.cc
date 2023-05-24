#include <math.h>
#include <stdio.h>

#include <cstdlib>
#include <iostream>

#include "lib/matmul.h"

#define BLK_SIZE 64
#define MAX_PRECISION_ERROR 0.01

#define A_ROW 1024
#define A_COLUMN 1024
#define B_ROW 1024
#define B_COLUMN 1024
#define C_ROW 1024
#define C_COLUMN 1024
#define NUM_THREAD 16

int8_t MAT_A_s8[A_ROW * A_COLUMN];
int8_t MAT_B_s8[B_ROW * B_COLUMN];
int8_t transpose_B_s8[B_ROW * B_COLUMN];
int8_t native_C_s8[C_ROW * C_COLUMN];
int8_t output_C_s8[C_ROW * C_COLUMN];

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

void initialize_matrix(int8_t A[], int size) {
    for (int i = 0; i < size; i++) {
        // A[i] = (rand() % 2) - 1;
        A[i] = (rand() % 2);
    }
}

using namespace matmul;

int main() {
    // initialize
    initialize_matrix(MAT_A_s8, A_ROW * A_COLUMN);
    initialize_matrix(MAT_B_s8, B_ROW * B_COLUMN);
    initialize_matrix(native_C_s8, C_ROW * C_COLUMN);

    MatmulOperator matmul_op = MatmulOperator();

    struct matmul_params params_int8;
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
    params_int8.C.qparams.zero_point = 0;
    params_int8.opt_params.blk_size = BLK_SIZE;
    params_int8.opt_params.num_thread = NUM_THREAD;

    // int8
    matmul_op.evaluate(MatmulOperator::INT8_BASELINE, &params_int8);

// ONEDNN
#ifdef ONEDNN_ENABLE
    initialize_matrix(output_C_s8, C_ROW * C_COLUMN);
    params_int8.C.int8_data_ptr = output_C_s8;
    matmul_op.evaluate(MatmulOperator::ONEDNN_INT8, &params_int8);
    if (!check_identical(native_C_s8, output_C_s8, C_ROW * C_COLUMN))
        printf("incorrect output from mat_mul_onednn_int8\n");
#endif

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
    // Debugging
    // dump_integer_array(MAT_A_s8, A_ROW * A_COLUMN);
    // dump_integer_array(MAT_B_s8, B_ROW * B_COLUMN);
    // dump_integer_array(native_C_s8, C_ROW * C_COLUMN);
    // dump_integer_array(output_C_s8, C_ROW * C_COLUMN);

    return 0;
}
