#ifndef MATMUL_H_
#define MATMUL_H_

#include <assert.h>
#include <math.h>
#include <pthread.h>
#include <stdio.h>
// #include <sys/time.h>
#include <immintrin.h>  // AVX intrinsic
#include <xmmintrin.h>  // intel SSE intrinsic

#include <iostream>
#include <string>

#include "matmul.h"

#define MAX_TRANSPOSE_BUFFER 2048 * 20480
#define RUNS 1

float transpose_tmp[MAX_TRANSPOSE_BUFFER];

namespace matmul {

void MatmulOperator::CHECK_MATRICES(const struct matrix *A, const struct matrix *B, const struct matrix *C) {
    assert(A->column == B->row);
    assert(C->column == B->column);
    assert(C->row == A->row);
}

inline void simd_mul_fp_128(const float *a, const float *b, float *c) {
    __m128 val = _mm_mul_ps(_mm_load_ps(a), _mm_load_ps(b));
    __m128 acc = _mm_add_ps(_mm_load_ps(c), val);
    _mm_store_ps(c, acc);
}

void MatmulOperator::naive_mat_mul(const struct matmul_params *params) {
    int i, j, k;
    const struct matrix *A = &params->A, *B = &params->B, *C = &params->C;
    float *data_A = A->data_ptr, *data_B = B->data_ptr, *data_C = C->data_ptr;
    CHECK_MATRICES(A, B, C);

    for (i = 0; i < C->row; i++)
        for (j = 0; j < C->column; j++) {
            float acc = 0;
            for (k = 0; k < A->column; k++) acc += data_A[i * A->column + k] * data_B[k * B->column + j];
            data_C[i * C->column + j] = acc;
        }
}

void MatmulOperator::mat_mul_unrolling(const struct matmul_params *params) {
    int i, j, k;

    const struct matrix *A = &params->A, *B = &params->B, *C = &params->C;
    float *data_A = A->data_ptr, *data_B = B->data_ptr, *data_C = C->data_ptr;
    CHECK_MATRICES(A, B, C);

    for (i = 0; i < C->row; i++)
        for (j = 0; j < C->column; j += 8) {
            float acc0 = 0;
            float acc1 = 0;
            float acc2 = 0;
            float acc3 = 0;
            float acc4 = 0;
            float acc5 = 0;
            float acc6 = 0;
            float acc7 = 0;
            for (k = 0; k < A->column; k += 4) {
                float Aik0 = data_A[i * A->column + k];
                float Aik1 = data_A[i * A->column + k + 1];
                float Aik2 = data_A[i * A->column + k + 2];
                float Aik3 = data_A[i * A->column + k + 3];

                acc0 += Aik0 * data_B[k * B->column + j];
                acc0 += Aik1 * data_B[(k + 1) * B->column + j];
                acc0 += Aik2 * data_B[(k + 2) * B->column + j];
                acc0 += Aik3 * data_B[(k + 3) * B->column + j];

                acc1 += Aik0 * data_B[k * B->column + j + 1];
                acc1 += Aik1 * data_B[(k + 1) * B->column + j + 1];
                acc1 += Aik2 * data_B[(k + 2) * B->column + j + 1];
                acc1 += Aik3 * data_B[(k + 3) * B->column + j + 1];

                acc2 += Aik0 * data_B[k * B->column + j + 2];
                acc2 += Aik1 * data_B[(k + 1) * B->column + j + 2];
                acc2 += Aik2 * data_B[(k + 2) * B->column + j + 2];
                acc2 += Aik3 * data_B[(k + 3) * B->column + j + 2];

                acc3 += Aik0 * data_B[k * B->column + j + 3];
                acc3 += Aik1 * data_B[(k + 1) * B->column + j + 3];
                acc3 += Aik2 * data_B[(k + 2) * B->column + j + 3];
                acc3 += Aik3 * data_B[(k + 3) * B->column + j + 3];

                acc4 += Aik0 * data_B[k * B->column + j + 4];
                acc4 += Aik1 * data_B[(k + 1) * B->column + j + 4];
                acc4 += Aik2 * data_B[(k + 2) * B->column + j + 4];
                acc4 += Aik3 * data_B[(k + 3) * B->column + j + 4];

                acc5 += Aik0 * data_B[k * B->column + j + 5];
                acc5 += Aik1 * data_B[(k + 1) * B->column + j + 5];
                acc5 += Aik2 * data_B[(k + 2) * B->column + j + 5];
                acc5 += Aik3 * data_B[(k + 3) * B->column + j + 5];

                acc6 += Aik0 * data_B[k * B->column + j + 6];
                acc6 += Aik1 * data_B[(k + 1) * B->column + j + 6];
                acc6 += Aik2 * data_B[(k + 2) * B->column + j + 6];
                acc6 += Aik3 * data_B[(k + 3) * B->column + j + 6];

                acc7 += Aik0 * data_B[k * B->column + j + 7];
                acc7 += Aik1 * data_B[(k + 1) * B->column + j + 7];
                acc7 += Aik2 * data_B[(k + 2) * B->column + j + 7];
                acc7 += Aik3 * data_B[(k + 3) * B->column + j + 7];
            }
            data_C[i * C->column + j] = acc0;
            data_C[i * C->column + j + 1] = acc1;
            data_C[i * C->column + j + 2] = acc2;
            data_C[i * C->column + j + 3] = acc3;
            data_C[i * C->column + j + 4] = acc4;
            data_C[i * C->column + j + 5] = acc5;
            data_C[i * C->column + j + 6] = acc6;
            data_C[i * C->column + j + 7] = acc7;
        }
}

void MatmulOperator::mat_mul_reordering(const struct matmul_params *params) {
    int i, j, k;
    float Aik;

    const struct matrix *A = &params->A, *B = &params->B, *C = &params->C;
    float *data_A = A->data_ptr, *data_B = B->data_ptr, *data_C = C->data_ptr;
    CHECK_MATRICES(A, B, C);

    for (i = 0; i < C->row; i++)
        for (j = 0; j < C->column; j++) data_C[i * C->column + j] = 0;

    for (i = 0; i < C->row; i++)
        for (k = 0; k < A->column; k++) {
            Aik = data_A[i * A->column + k];
            for (j = 0; j < C->column; j++) {
                data_C[i * C->column + j] += Aik * data_B[k * B->column + j];
            }
        }
}

void MatmulOperator::mat_mul_tiling(const struct matmul_params *params) {
    int i, j, k, ti, tj, tk;
    float Aik;

    int BLK_SIZE = params->opt_params.blk_size;

    const struct matrix *A = &params->A, *B = &params->B, *C = &params->C;
    float *data_A = A->data_ptr, *data_B = B->data_ptr, *data_C = C->data_ptr;
    CHECK_MATRICES(A, B, C);
    assert(C->row % BLK_SIZE == 0);
    assert(A->column % BLK_SIZE == 0);
    assert(C->column % BLK_SIZE == 0);

    for (i = 0; i < C->row; i++)
        for (j = 0; j < C->column; j++) data_C[i * C->column + j] = 0;

    for (ti = 0; ti < C->row; ti += BLK_SIZE) {
        for (tk = 0; tk < A->column; tk += BLK_SIZE) {
            for (tj = 0; tj < C->column; tj += BLK_SIZE) {
                for (i = ti; i < ti + BLK_SIZE; i++)
                    for (k = tk; k < tk + BLK_SIZE; k++) {
                        Aik = data_A[i * A->column + k];
                        for (j = tj; j < tj + BLK_SIZE; j++) {
                            data_C[i * C->column + j] += Aik * data_B[k * B->column + j];
                        }
                    }
            }
        }
    }
}

/* This function assume legal matrices */
void *thread_func(void *args) {
    struct thread_args *mat_args = (struct thread_args *)args;
    const struct matrix *A = mat_args->A;
    const struct matrix *B = mat_args->B;
    const struct matrix *C = mat_args->C;
    float *data_A = A->data_ptr, *data_B = B->data_ptr, *data_C = C->data_ptr;
    int start_i = mat_args->start_i, end_i = mat_args->end_i;

    for (int i = start_i; i < end_i; i++)
        for (int j = 0; j < C->column; j++) {
            float acc = 0;
            for (int k = 0; k < A->column; k++) acc += data_A[i * A->column + k] * data_B[k * B->column + j];
            data_C[i * C->column + j] = acc;
        }

    return NULL;
}

void MatmulOperator::mat_mul_multithreading(const struct matmul_params *params) {
    int j, num_thread = params->opt_params.num_thread;

    const struct matrix *A = &params->A, *B = &params->B, *C = &params->C;
    CHECK_MATRICES(A, B, C);
    assert(num_thread != 0);
    assert(C->row % num_thread == 0);

    pthread_t thread_pool[num_thread];
    struct thread_args threads_args[num_thread];

    // Thread creation
    for (j = 0; j < num_thread; j++) {
        threads_args[j].start_i = j * (C->row / num_thread);
        threads_args[j].end_i = (j + 1) * (C->row / num_thread);
        threads_args[j].A = A;
        threads_args[j].B = B;
        threads_args[j].C = C;
        pthread_create(&thread_pool[j], NULL, thread_func, &threads_args[j]);
    }
    // Join threads
    for (j = 0; j < num_thread; j++) {
        pthread_join(thread_pool[j], NULL);
    }
}

void MatmulOperator::mat_mul_transpose(const struct matmul_params *params) {
    int i, j, k;

    const struct matrix *A = &params->A, *B = &params->B, *C = &params->C;
    float *data_A = A->data_ptr, *data_B = B->data_ptr, *data_C = C->data_ptr;
    CHECK_MATRICES(A, B, C);

    // transpose the B
    for (i = 0; i < B->column; i++)
        for (j = 0; j < B->row; j++) transpose_tmp[i * B->row + j] = data_B[j * B->column + i];

    for (i = 0; i < C->row; i++)
        for (j = 0; j < C->column; j++) {
            float acc = 0;
            for (k = 0; k < A->column; k++) acc += data_A[i * A->column + k] * transpose_tmp[j * B->row + k];
            data_C[i * C->column + j] = acc;
        }
}

void *mat_mul_transposed_fastover_column_func(void *args) {
    int i, j, k;
    struct thread_args *mat_args = (struct thread_args *)args;
    const struct matrix *A = mat_args->A;
    const struct matrix *B = mat_args->B;
    const struct matrix *C = mat_args->C;
    float *data_A = A->data_ptr, *data_B = B->data_ptr, *data_C = C->data_ptr;
    int start_i = mat_args->start_i, end_i = mat_args->end_i;

    __m256 zero256 = _mm256_setzero_ps();
    for (i = 0; i < C->row; i++) {
        for (j = start_i; j + 1 < end_i; j += 2) {
            __m256 acc = zero256, acc1 = zero256;
            __m256 *A256 = (__m256 *)&data_A[i * A->column];
            __m256 *B256 = (__m256 *)&data_B[j * B->column];
            __m256 *B256_1 = (__m256 *)&data_B[(j + 1) * B->column];
            for (k = 0; k < A->column; k += 8) {
                __m256 Aik = _mm256_load_ps((const float *)A256++);
                __m256 Bjk = _mm256_load_ps((const float *)B256++);
                __m256 Bj1k = _mm256_load_ps((const float *)B256_1++);
                acc = _mm256_add_ps(acc, _mm256_mul_ps(Aik, Bjk));
                acc1 = _mm256_add_ps(acc1, _mm256_mul_ps(Aik, Bj1k));
            }
            float *ptr = (float *)&acc;
            data_C[i * C->column + j] = ptr[0] + ptr[1] + ptr[2] + ptr[3] + ptr[4] + ptr[5] + ptr[6] + ptr[7];
            ptr = (float *)&acc1;
            data_C[i * C->column + j + 1] = ptr[0] + ptr[1] + ptr[2] + ptr[3] + ptr[4] + ptr[5] + ptr[6] + ptr[7];
        }
        // leftover
        if (j < end_i) {
            __m256 acc = zero256;
            for (k = 0; k < A->column; k += 8) {
                __m256 Aik = _mm256_load_ps(&data_A[i * A->column + k]);
                __m256 Bjk = _mm256_load_ps(&data_B[j * B->column + k]);
                acc = _mm256_add_ps(acc, _mm256_mul_ps(Aik, Bjk));
            }
            float *ptr = (float *)&acc;
            data_C[i * C->column + j] = ptr[0] + ptr[1] + ptr[2] + ptr[3] + ptr[4] + ptr[5] + ptr[6] + ptr[7];
            j++;
        }
    }

    return NULL;
}

void MatmulOperator::mat_mul_transposed_fastover_column(const struct matmul_params *params) {
    int i, j, k;

    int num_thread = params->opt_params.num_thread;
    const struct matrix *A = &params->A, *B = &params->B, *C = &params->C;
    float *data_A = A->data_ptr, *data_B = B->data_ptr, *data_C = C->data_ptr;

    assert(A->column % 8 == 0);

    if (num_thread > C->column) num_thread = C->column;

    pthread_t thread_pool[num_thread];
    struct thread_args threads_args[num_thread];

    // Thread creation
    for (j = 0; j < num_thread; j++) {
        threads_args[j].start_i = j * (C->column / num_thread);
        threads_args[j].end_i = (j + 1) * (C->column / num_thread);
        threads_args[j].blk_size = params->opt_params.blk_size;
        threads_args[j].A = A;
        threads_args[j].B = B;
        threads_args[j].C = C;
        pthread_create(&thread_pool[j], NULL, mat_mul_transposed_fastover_column_func, &threads_args[j]);
    }
    // Join threads
    for (j = 0; j < num_thread; j++) {
        pthread_join(thread_pool[j], NULL);
    }
}

void MatmulOperator::mat_mul_transpose_simd(const struct matmul_params *params) {
    int i, j, k;

    const struct matrix *A = &params->A, *B = &params->B, *C = &params->C;
    float *data_A = A->data_ptr, *data_B = B->data_ptr, *data_C = C->data_ptr;
    CHECK_MATRICES(A, B, C);

    // transpose the B
    for (i = 0; i < B->column; i++)
        for (j = 0; j < B->row; j++) transpose_tmp[i * B->row + j] = data_B[j * B->column + i];

    for (i = 0; i < C->row; i++)
        for (j = 0; j < C->column; j++) {
            float accumulators[4] = {};
            for (k = 0; k < A->column; k += 4)
                simd_mul_fp_128(&data_A[i * A->column + k], &transpose_tmp[j * B->row + k], accumulators);
            data_C[i * C->column + j] = accumulators[0] + accumulators[1] + accumulators[2] + accumulators[3];
        }
}

float interval_to_ms(struct timeval *start, struct timeval *end) {
    float us_seconds = (end->tv_sec - start->tv_sec) * 1000000 + (end->tv_usec - start->tv_usec);
    return us_seconds / 1000;
}

void MatmulOperator::evaluate(IMP_TYPE type, const struct matmul_params *params) {
    struct timeval start, end;
    int ms;
    std::string function_name;

    gettimeofday(&start, NULL);
    // choose implementation
    switch (type) {
        case NAIVE:
            function_name = "naive_mat_mul";
            for (int i = 0; i < RUNS; i++) this->naive_mat_mul(params);
            break;
        case UNROLL:
            function_name = "mat_mul_unrolling";
            for (int i = 0; i < RUNS; i++) this->mat_mul_unrolling(params);
            break;
        case REORDER:
            function_name = "mat_mul_reordering";
            for (int i = 0; i < RUNS; i++) this->mat_mul_reordering(params);
            break;
        case TILING:
            function_name = "mat_mul_tiling";
            for (int i = 0; i < RUNS; i++) this->mat_mul_tiling(params);
            break;
        case MULTITHREAD:
            function_name = "mat_mul_multithreading";
            for (int i = 0; i < RUNS; i++) this->mat_mul_multithreading(params);
            break;
        case TRANSPOSE:
            function_name = "mat_mul_transpose";
            for (int i = 0; i < RUNS; i++) this->mat_mul_transpose(params);
            break;
        case TRANSPOSE_SIMD:
            function_name = "mat_mul_transpose_simd";
            for (int i = 0; i < RUNS; i++) this->mat_mul_transpose_simd(params);
            break;
        case CUDA:
            function_name = "mat_mul_cuda";
#ifdef CUDA_ENABLE
            for (int i = 0; i < RUNS; i++) this->mat_mul_cuda(params);
#else
            fprintf(stderr, "CUDA not enable!\n");
            exit(-1);
#endif
            break;
        case FAST:
            function_name = "mat_mul_fast";
            for (int i = 0; i < RUNS; i++) this->mat_mul_fast(params);
            break;
        case ONEDNN_FP32:
            function_name = "mat_mul_onednn";
#ifdef ONEDNN_ENABLE
            for (int i = 0; i < RUNS; i++) this->mat_mul_onednn(params);
#else
            fprintf(stderr, "ONEDNN not enable!\n");
            exit(-1);
#endif
            break;
        case ONEDNN_INT8:
            function_name = "mat_mul_onednn_int8";
#ifdef ONEDNN_ENABLE
            for (int i = 0; i < RUNS; i++) this->mat_mul_onednn_int8(params);
#else
            fprintf(stderr, "ONEDNN not enable!\n");
            exit(-1);
#endif
            break;
        case INT8_BASELINE:
            function_name = "naive_mat_mul_int8";
            for (int i = 0; i < RUNS; i++) this->naive_mat_mul_int8(params);
            break;
        case INT8_AVX:
            function_name = "mat_mul_avx_int8";
            for (int i = 0; i < RUNS; i++) this->mat_mul_avx_int8(params);
            break;
        case INT8_AVX_FAST:
            function_name = "mat_mul_avx_int8_fast";
            for (int i = 0; i < RUNS; i++) this->mat_mul_avx_int8_fast(params);
            break;
        case INT8_AVX_FAST_2x2:
            function_name = "mat_mul_avx_int8_fast_2x2";
            for (int i = 0; i < RUNS; i++) this->mat_mul_avx_int8_fast_2x2(params);
            break;
        case INT8_AVX_FAST_2x2_32UNROLL:
            function_name = "mat_mul_avx_int8_fast_2x2_32unroll";
            for (int i = 0; i < RUNS; i++) this->mat_mul_avx_int8_fast_2x2_32unroll(params);
            break;
        case INT8_AVX_FAST_2x2_OMP:
            function_name = "mat_mul_avx_int8_fast_2x2_omp";
            for (int i = 0; i < RUNS; i++) this->mat_mul_avx_int8_fast_2x2_omp(params);
            break;
        default:
            break;
    }
    gettimeofday(&end, NULL);
    ms = interval_to_ms(&start, &end);
    float GOPS =
        (float)((float)params->C.column * (float)params->C.row * (float)params->B.row) * 2 / (1000000000) * RUNS;
    std::cout << function_name << ": " << ms << " ms, GOPS/s:" << GOPS / ((float)ms / 1000) << std::endl;
}

void *fast_thread_func(void *args) {
    struct thread_args *mat_args = (struct thread_args *)args;
    const struct matrix *A = mat_args->A;
    const struct matrix *B = mat_args->B;
    const struct matrix *C = mat_args->C;
    float *data_A = A->data_ptr, *data_B = B->data_ptr, *data_C = C->data_ptr;
    int start_i = mat_args->start_i, end_i = mat_args->end_i;

    int BLK_SIZE = mat_args->blk_size;
    assert((end_i - start_i) % BLK_SIZE == 0);
    assert(A->column % BLK_SIZE == 0);
    assert(C->column % BLK_SIZE == 0);
    assert(BLK_SIZE % 4 == 0);

    for (int ti = start_i; ti < end_i; ti += BLK_SIZE) {
        for (int tj = 0; tj < C->column; tj += BLK_SIZE) {
            for (int i = ti; i < ti + BLK_SIZE; i++)
                for (int j = tj; j < tj + BLK_SIZE; j += 4) {
                    float acc0[4] = {}, acc1[4] = {}, acc2[4] = {}, acc3[4] = {};
                    __m128 *acc0_fp_128 = (__m128 *)acc0;
                    __m128 *acc1_fp_128 = (__m128 *)acc1;
                    __m128 *acc2_fp_128 = (__m128 *)acc2;
                    __m128 *acc3_fp_128 = (__m128 *)acc3;

                    for (int k = 0; k < A->column; k += 4) {
                        __m128 Aik_Aik3 = _mm_load_ps(&data_A[i * A->column + k]);
                        __m128 val;
                        val = _mm_mul_ps(Aik_Aik3, _mm_load_ps(&data_B[j * B->column + k]));
                        *acc0_fp_128 = _mm_add_ps(*acc0_fp_128, val);

                        val = _mm_mul_ps(Aik_Aik3, _mm_load_ps(&data_B[(j + 1) * B->column + k]));
                        *acc1_fp_128 = _mm_add_ps(*acc1_fp_128, val);

                        val = _mm_mul_ps(Aik_Aik3, _mm_load_ps(&data_B[(j + 2) * B->column + k]));
                        *acc2_fp_128 = _mm_add_ps(*acc2_fp_128, val);

                        val = _mm_mul_ps(Aik_Aik3, _mm_load_ps(&data_B[(j + 3) * B->column + k]));
                        *acc3_fp_128 = _mm_add_ps(*acc3_fp_128, val);
                    }
                    data_C[i * C->column + j] = acc0[0] + acc0[1] + acc0[2] + acc0[3];
                    data_C[i * C->column + j + 1] = acc1[0] + acc1[1] + acc1[2] + acc1[3];
                    data_C[i * C->column + j + 2] = acc2[0] + acc2[1] + acc2[2] + acc2[3];
                    data_C[i * C->column + j + 3] = acc3[0] + acc3[1] + acc3[2] + acc3[3];
                }
        }
    }

    return NULL;
}

void MatmulOperator::mat_mul_fast(const struct matmul_params *params) {
    int j, num_thread = params->opt_params.num_thread;

    const struct matrix *A = &params->A, *B = &params->B, *C = &params->C;

    assert(A->column == B->column);
    assert(C->column == B->row);
    assert(C->row == A->row);
    assert(num_thread != 0);
    assert(C->row % num_thread == 0);

    pthread_t thread_pool[num_thread];
    struct thread_args threads_args[num_thread];

    // Thread creation
    for (j = 0; j < num_thread; j++) {
        threads_args[j].start_i = j * (C->row / num_thread);
        threads_args[j].end_i = (j + 1) * (C->row / num_thread);
        threads_args[j].blk_size = params->opt_params.blk_size;
        threads_args[j].A = A;
        threads_args[j].B = B;
        threads_args[j].C = C;
        pthread_create(&thread_pool[j], NULL, fast_thread_func, &threads_args[j]);
    }
    // Join threads
    for (j = 0; j < num_thread; j++) {
        pthread_join(thread_pool[j], NULL);
    }
}

}  // namespace matmul

#endif  // MATMUL_H_
