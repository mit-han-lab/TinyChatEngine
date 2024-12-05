#ifndef MATMUL_OPERATOR_CUDA_H
#define MATMUL_OPERATOR_CUDA_H

#include "matmul.h"
#include <iostream>

namespace matmul {

class MatmulOperatorCUDA : public MatmulOperator {
   public:
    void mat_mul_accelerator_transposed_fastover_column(const struct matmul_params* params) override;

    // int8 operations
    void mat_mul_accelerator_int8_fast_32unroll_over_column(const struct matmul_params* params) override;
    void mat_mul_accelerator_int8_fast_2x2_32unroll(const struct matmul_params* params) override;
    void mat_mul_accelerator_int8_fast_2x2_32unroll_nobias(const struct matmul_params* params) override;
    void mat_mul_accelerator_int8_fast_2x2_32unroll_nobias_batch(const struct matmul_params* params) override;
    void mat_mul_accelerator_int8_fast_2x2_32unroll_nobias_ofp32(const struct matmul_params* params) override;
    void mat_mul_accelerator_int8_fast_2x2_32unroll_nobias_ofp32_batch(const struct matmul_params* params) override;
    void mat_mul_accelerator_int8_fast_2x2_32unroll_bfp32_ofp32(const struct matmul_params* params) override;
    void mat_mul_accelerator_int8_fast_2x2_32unroll_bfp32_ofp32_over_column(const struct matmul_params* params) override;

    void mat_mul_accelerator_int4_fast(const struct matmul_params* params) override;
    void mat_mul_accelerator_int4_fast_no_offset(const struct matmul_params* params) override;

    void gemv_forward_cuda(const struct matmul_params* params) override;
    void naive_mat_mul_fp16_int4(const struct matmul_params* params) override;
};

// Declaring as static to prevent linker errors due to both cc and cu files
static inline MatmulOperator& CreateMatmulOperatorCUDA() {
    static MatmulOperatorCUDA instance;
    return instance;
}

}  // namespace matmul

#endif 
