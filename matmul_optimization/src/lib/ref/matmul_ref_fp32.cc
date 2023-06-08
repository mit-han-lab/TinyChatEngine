#include <assert.h>
#include <pthread.h>
#include <stdio.h>

#include <cmath>
#include <cstdlib>

#include "../matmul.h"

namespace matmul {
void fp32_ref_matmul(const struct matmul_params *params) {
    const struct matrix *A = &params->A, *B = &params->B, *C = &params->C;
    float *data_A = A->data_ptr, *data_B = B->data_ptr, *data_C = C->data_ptr;

    assert(A->column == B->column);
    assert(C->row == A->row);
    assert(C->column == B->row);
    int m = A->row, n = B->row, k = A->column;

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float acc = 0;
            for (int kk = 0; kk < k; k++) {
                acc += data_A[i * A->column + kk] * data_B[j * B->row + kk];
            }
            acc = acc;
            data_C[i * C->column + j] = acc;
        }
    }
}

void MatmulOperator::mat_mul_accelerator_transposed_fastover_column(const struct matmul_params *params) {
    fp32_ref_matmul(params);
}

}  // namespace matmul
