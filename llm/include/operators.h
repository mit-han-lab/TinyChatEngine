#ifndef OPERATORS_H
#define OPERATORS_H
#include <cassert>

#include "common.h"
#include "matmul.h"

#define BLK_SIZE 16
// #define NUM_THREAD 8
extern int NUM_THREAD;

// include all ops
#include "ops/BMM_F32T.h"
#include "ops/BMM_S8T_S8N_F32T.h"
#include "ops/BMM_S8T_S8N_S8T.h"
#include "ops/Embedding.h"
#include "ops/LayerNorm.h"
#include "ops/LayerNormQ.h"
#include "ops/LlamaRMSNorm.h"
#include "ops/RotaryPosEmb.h"
#include "ops/W8A8B8O8Linear.h"
#include "ops/W8A8B8O8LinearReLU.h"
#include "ops/W8A8BFP32OFP32Linear.h"
#include "ops/arg_max.h"
#include "ops/linear.h"

void softmax(const Matrix3D<float> &input, Matrix3D<float> &output, int dim);
void batch_Add(const Matrix3D<float> &input, const Matrix3D<float> &input2, Matrix3D<float> &output);
template <typename T>
void linear(Matrix3D<T> &a, Matrix3D<T> &b, Matrix3D<T> &c);

#ifdef QM_CUDA
#include "ops/cuda/BMM_F16T.cuh"
#include "ops/cuda/Embedding.cuh"
#include "ops/cuda/LlamaRMSNorm.cuh"
#include "ops/cuda/RotaryPosEmb.cuh"

__global__ void batch_Add_float(const Matrix3D<float> input, const Matrix3D<float> input2, Matrix3D<float> output);
__global__ void batch_Add_cuda(const Matrix3D<float16_t> input, const Matrix3D<float16_t> input2,
                               Matrix3D<float16_t> output);
__global__ void batch_Add_cuda_half2(Matrix3D<float16_t> input, Matrix3D<float16_t> input2, Matrix3D<float16_t> output);
__global__ void softmax_float(Matrix3D<float> input, Matrix3D<float> output);
__global__ void softmax_cuda(Matrix3D<float16_t> input, Matrix3D<float16_t> output);
#endif

#ifdef QM_METAL
#include "ops/metal/BMM_F16T.cuh"
#include "ops/metal/Embedding.cuh"
#include "ops/metal/LlamaRMSNorm.cuh"
#include "ops/metal/RotaryPosEmb.cuh"

void batch_Add_metal(const Matrix3D<float> input, const Matrix3D<float> input2, Matrix3D<float> output);
void softmax_metal(Matrix3D<float16_t> input, Matrix3D<float16_t> output);
#endif

#endif  // OPERATORS_H
