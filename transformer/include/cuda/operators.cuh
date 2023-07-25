#ifndef OPERATORS_CUH
#define OPERATORS_CUH
#include <cassert>

// include all ops
#include "../ops/cuda/RotaryPosEmb.cuh"
#include "../ops/cuda/BMM_F16T.cuh"
#include "../ops/cuda/LlamaRMSNorm.cuh"
#include "../ops/cuda/Embedding.cuh"
// #include "linear.cuh"
#include "operators.h"

__global__ void batch_Add_float(const Matrix3D<float> input, const Matrix3D<float> input2, Matrix3D<float> output);
__global__ void batch_Add_cuda(const Matrix3D<float16_t> input, const Matrix3D<float16_t> input2, Matrix3D<float16_t> output);
__global__ void softmax_float(Matrix3D<float> input, Matrix3D<float> output);
__global__ void softmax_cuda(Matrix3D<float16_t> input, Matrix3D<float16_t> output);

#endif  // OPERATORS_H
