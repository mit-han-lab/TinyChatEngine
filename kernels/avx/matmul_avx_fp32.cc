#include <assert.h>
#include <immintrin.h>  // AVX intrinsic
#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <xmmintrin.h>  // intel SSE intrinsic

#include "../matmul.h"

namespace matmul {

inline void simd_mul_fp_128(const float *a, const float *b, float *c) {
    __m128 val = _mm_mul_ps(_mm_load_ps(a), _mm_load_ps(b));
    __m128 acc = _mm_add_ps(_mm_load_ps(c), val);
    _mm_store_ps(c, acc);
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
            __m256 *B256 = (__m256 *)&data_B[j * B->row];
            __m256 *B256_1 = (__m256 *)&data_B[(j + 1) * B->row];
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
                __m256 Bjk = _mm256_load_ps(&data_B[j * B->row + k]);
                acc = _mm256_add_ps(acc, _mm256_mul_ps(Aik, Bjk));
            }
            float *ptr = (float *)&acc;
            data_C[i * C->column + j] = ptr[0] + ptr[1] + ptr[2] + ptr[3] + ptr[4] + ptr[5] + ptr[6] + ptr[7];
            j++;
        }
    }

    return NULL;
}

void MatmulOperator::mat_mul_accelerator_transposed_fastover_column(const struct matmul_params *params) {
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

}  // namespace matmul
