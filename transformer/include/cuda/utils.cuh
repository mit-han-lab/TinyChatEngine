#ifndef UTILS_CUH
#define UTILS_CUH

#include <cstdlib>
#include <fstream>
#include <typeinfo>
#include <math.h>

// #if defined(__CUDACC__)
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
// #endif

// #include "half.hpp"  // Third-party header

// ////// TODO: Fix this
// #if defined(__CUDACC__)
//     #include <cuda.h>
//     #include <cuda_fp16.h>
//     #include <cuda_runtime.h>
//     typedef half float16_t;
// #elif defined(__ARM_NEON)
//     typedef __fp16 float16_t;
// #elif defined(__x86_64__)
//     printf("x86_64 does not natively support fp16, so we use `half_float` library to support fp16 through software-based solution.\n");
//     typedef half_float::half float16_t;
// #else
//     printf("Unsupported platform (we only support CUDA, Arm, and x86_64). Using uint16_t as float16_t.\n");
//     typedef uint16_t float16_t;
// #endif

bool check_two_equal_cpu_gpu(half* array, half* array2, int size, float error);
bool check_two_equal_float_half(float* array, half* array2, int size);

template <typename T>
void allocate_aligned_memory_gpu(T*& ptr, size_t size);

__global__ void float2half(float* floatArray, half* halfArray, int N);
__global__ void half2float(half* halfArray, float* floatArray, int N);
__global__ void half2float_merge_k_iters(half *halfArray, float *floatArray, int N, int split_k_iters);

#endif
