#ifndef MATMUL_OPERATOR_MKL_H
#define MATMUL_OPERATOR_MKL_H

#include "matmul.h"
#include <iostream>

namespace matmul {

class MatmulOperatorMKL : public MatmulOperator {
   public:
    void mat_mul_accelerator_transposed_fastover_column(const struct matmul_params* params) override;
    void mat_mul_accelerator_transposed_fastover_column_bias(const struct matmul_params* params) override;

    // int8 operations
    void mat_mul_accelerator_int8_fast_32unroll_over_column(const struct matmul_params* params) override;
    void mat_mul_accelerator_int8_fast_2x2_32unroll(const struct matmul_params* params) override;
    void mat_mul_accelerator_int8_fast_2x2_32unroll_nobias(const struct matmul_params* params) override;
    void mat_mul_accelerator_int8_fast_2x2_32unroll_nobias_batch(const struct matmul_params* params) override;
    void mat_mul_accelerator_int8_fast_2x2_32unroll_nobias_ofp32(const struct matmul_params* params) override;
    void mat_mul_accelerator_int8_fast_2x2_32unroll_nobias_ofp32_batch(const struct matmul_params* params) override;
    void mat_mul_accelerator_int8_fast_2x2_32unroll_bfp32_ofp32(const struct matmul_params* params) override;
    void mat_mul_accelerator_int8_fast_2x2_32unroll_bfp32_ofp32_over_column(const struct matmul_params* params) override;

    void mat_mul_accelerator_int8_int4_fast_no_offset(struct matmul_params* params) override;

    void mat_mul_accelerator_int4_fast(const struct matmul_params* params) override;
    void mat_mul_accelerator_int4_fast_no_offset(const struct matmul_params* params) override;
};

inline MatmulOperator& CreateMatmulOperatorMKL() {
    static MatmulOperatorMKL instance;
    return instance;
}

}  // namespace matmul

#endif 
