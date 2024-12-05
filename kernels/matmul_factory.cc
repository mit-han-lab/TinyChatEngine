#include "matmul.h"
#include "avx/matmul_avx.h"
#include "mkl/matmul_mkl.h"
#include "cuda/matmul_cuda.h"
#include "neon/matmul_neon.h"
#include "ref/matmul_ref.h"

namespace matmul {

// Declare external factory functions for each implementation
MatmulOperator& CreateMatmulOperatorMKL();
MatmulOperator& CreateMatmulOperatorAVX();
MatmulOperator& CreateMatmulOperatorCUDA();
MatmulOperator& CreateMatmulOperatorNeon();
MatmulOperator& CreateMatmulOperatorRef();

MatmulOperator& CreateMatmulOperator() {
#ifdef QM_CUDA
    return CreateMatmulOperatorCUDA();
#elif defined(QM_MKL)
    return CreateMatmulOperatorMKL();
#elif defined(QM_ARM)
    return CreateMatmulOperatorNeon();
#elif defined(QM_x86)
    return CreateMatmulOperatorAVX(); // Default to AVX
#else
    return CreateMatmulOperatorRef();
#endif
}

}