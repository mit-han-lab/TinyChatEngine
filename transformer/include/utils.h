#ifndef UTILS_H
#define UTILS_H

#include <math.h>

#include <cstdlib>
#include <fstream>
#include <typeinfo>

#include "profiler.h"

#include "half.hpp"  // Third-party header

#define STATS_START(x) Profiler::getInstance().start(x)
#define STATS_FLOPS(x, y) Profiler::getInstance().start(x, y)
#define STATS_END(x) Profiler::getInstance().stop(x)

#ifdef PROFILER
#define PROFILE_START(x) Profiler::getInstance().start(x)
#define PROFILE_START_FLOPS(x, y) Profiler::getInstance().start(x, y)
#define PROFILE_END(x) Profiler::getInstance().stop(x)
#else
#define PROFILE_START(x)
#define PROFILE_START_FLOPS(x, y)
#define PROFILE_END(x)
#endif

#define MAX_SQ_ERROR_MAX 5e-6
#define ERROR_MAX 1e-9
#define INT_ERROR_MAX 1e-5

template <typename T>
void read_to_array(const char* path, T* array, int size);

template <typename T>
bool check_two_equal(T* array, T* array2, int size);

template <>
bool check_two_equal(int8_t* array, int8_t* array2, int size);

bool check_two_equal(int8_t* array, int8_t* array2, int size, float error);

bool check_two_equal(float* array, float* array2, int size, float error);
bool check_two_exact_equal(int8_t* array, int8_t* array2, int size);
void print_MSE_max_diff(float* a, float* a2, int size);

void print_first_k_elelment(std::string name, const int8_t* arr, int k, int start_idx = 0);
void print_first_k_elelment(std::string name, const int32_t* arr, int k, int start_idx = 0);
void print_first_k_elelment(std::string name, const float* arr, int k, int start_idx = 0);

#ifdef QM_METAL
template <typename T>
void allocate_aligned_memory(T*& ptr, size_t size);
#else
template <typename T>
void allocate_aligned_memory(T*& ptr, size_t size);
#endif

void deallocate_memory(void* ptr);

#ifdef QM_CUDA
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if(err != cudaSuccess) { \
            printf("Error: %s:%d, ", __FILE__, __LINE__);                 \
            printf("code: %d, reason: %s\n", err,                       \
                   cudaGetErrorString(err));                            \
            throw std::runtime_error(std::string("CUDA error calling \"") + #call + "\", code is " + std::to_string(err)); \
        } \
    } while(0)

extern half *split_8_buffer;

void read_to_array_half(const char* path, half* array, int size);

bool check_two_equal_cpu_gpu(half_float::half* array, half* array2, int size, float error);
bool check_two_equal_float_half(float* array, half* array2, int size);
bool check_two_equal_half_half(half* array, half* array2, int size);

template <typename T>
void allocate_aligned_memory_gpu(T*& ptr, size_t size);

template <typename T>
void free_aligned_memory_gpu(T*& ptr);

__global__ void float2half(float* floatArray, half* halfArray, int N);
__global__ void half2float(half* halfArray, float* floatArray, int N);
__global__ void half2float_merge_k_iters(half *halfArray, float *floatArray, int N, int split_k_iters);
__global__ void merge_k_iters(half *input, half *output, int N, int split_k_iters);
#endif

#endif
