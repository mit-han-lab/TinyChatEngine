#ifndef UTILS_H
#define UTILS_H

#include <cstdlib>
#include <fstream>
#include <typeinfo>
#include <math.h>

#include "profiler.h"

// #if defined(__CUDACC__)
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
// #endif

#include "half.hpp"  // Third-party header

////// TODO: Fix this
#if defined(__CUDACC__)
    typedef half float16_t;
#elif defined(__ARM_NEON)
    typedef __fp16 float16_t;
#elif defined(__x86_64__)
    printf("x86_64 does not natively support fp16, so we use `half_float` library to support fp16 through software-based solution.\n");
    typedef half_float::half float16_t;
#else
    printf("Unsupported platform (we only support CUDA, Arm, and x86_64). Using uint16_t as float16_t.\n");
    typedef uint16_t float16_t;
#endif


#define QK 32

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

bool check_two_equal(float* array, float* array2, int size, float error);

bool check_two_equal_cpu_gpu(float16_t* array, half* array2, int size, float error);

bool check_two_exact_equal(int8_t* array, int8_t* array2, int size);
void print_MSE_max_diff(float* a, float* a2, int size);

void print_first_k_elelment(std::string name, const int8_t* arr, int k, int start_idx = 0);
void print_first_k_elelment(std::string name, const int32_t* arr, int k, int start_idx = 0);
void print_first_k_elelment(std::string name, const float* arr, int k, int start_idx = 0);

template <typename T>
void allocate_aligned_memory(T*& ptr, size_t size);

template <typename T>
void allocate_aligned_memory_gpu(T*& ptr, size_t size);

#endif
