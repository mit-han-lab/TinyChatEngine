#include "utils.h"

#include <stdio.h>

#include <cassert>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>

half *split_8_buffer = nullptr;

void read_to_array_half(const char* path, half* array, int size) {
    std::ifstream infile(path, std::ios::binary | std::ios::in);
    if (infile.fail()) {
        std::cout << strerror(errno) << ": " << path << std::endl;
        throw("Expected error...");
    } else {
        infile.read(reinterpret_cast<char*>(array), size * sizeof(half));
        infile.close();
    }
}

struct max_error_info {
    int idx;
    float a1, a2;
};

bool check_two_equal_cpu_gpu(half_float::half* array, half* array2, int size, float error) {
    float sq_diff = 0;
    float max_sqdiff = 0;
    struct max_error_info error_info;

    for (int i = 0; i < size; i++) {
        float diff = static_cast<float>(array[i]) - __half2float(array2[i]);

        sq_diff += diff * diff;
        if (diff * diff > max_sqdiff) {
            max_sqdiff = diff * diff;
            error_info.idx = i;
            error_info.a1 = static_cast<float>(array[i]);
            error_info.a2 = __half2float(array2[i]);
        }
    }
    if ((sq_diff / size) > error) {
        std::cout << "MSE:" << sq_diff / size << ", MAX SQ diff:" << max_sqdiff;
        std::cout << "@:" << error_info.idx << ",a1:" << error_info.a1 << ",a2:" << error_info.a2 << std::endl;
        return false;
    }
    return true;
}

bool check_two_equal_float_half(float* array, half* array2, int size) {
    float sq_diff = 0;
    float max_sqdiff = 0;
    for (int i = 0; i < size; i++) {
        float diff = (float)array[i] - __half2float(array2[i]);
        sq_diff += diff * diff;
        if (diff * diff > max_sqdiff) max_sqdiff = diff * diff;
        if (sqrt(max_sqdiff) > MAX_SQ_ERROR_MAX) {
            std::cout << "i:" << i << ",max_sqdiff:" << sqrt(max_sqdiff) << ", array[i]:";
            std::cout << static_cast<float>(array[i]) << ", array2[i]:" << __half2float(array2[i]) << std::endl;
            return false;
        }
    }
    if ((sq_diff / size) > ERROR_MAX) {
        std::cout << "MSE:" << sq_diff / size << ", MAX SQ diff:" << max_sqdiff << std::endl;
        return false;
    }
    return true;
}

bool check_two_equal_half_half(half* array, half* array2, int size) {
    float sq_diff = 0;
    float max_sqdiff = 0;
    for (int i = 0; i < size; i++) {
        float diff = __half2float(array[i]) - __half2float(array2[i]);
        sq_diff += diff * diff;
        if (diff * diff > max_sqdiff) max_sqdiff = diff * diff;
        if (sqrt(max_sqdiff) > MAX_SQ_ERROR_MAX) {
            std::cout << "i:" << i << ",max_sqdiff:" << sqrt(max_sqdiff) << ", array[i]:";
            std::cout << __half2float(array[i]) << ", array2[i]:" << __half2float(array2[i]) << std::endl;
            return false;
        }
    }
    if ((sq_diff / size) > ERROR_MAX) {
        std::cout << "MSE:" << sq_diff / size << ", MAX SQ diff:" << max_sqdiff << std::endl;
        return false;
    }
    return true;
}

template <typename T>
void allocate_aligned_memory_gpu(T*& ptr, size_t size) {
    // Allocate unified memory
    CHECK_CUDA(cudaMallocManaged((void**)&ptr, size));
}

template <typename T>
void free_aligned_memory_gpu(T*& ptr) {
    if (ptr) {
        CHECK_CUDA(cudaFree(ptr));
        ptr = nullptr;
    }
}

__global__ void float2half(float* floatArray, half* halfArray, int N) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < N) {
        halfArray[index] = __float2half(floatArray[index]);
    }
}

__global__ void half2float(half* halfArray, float* floatArray, int N) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < N) {
        floatArray[index] = __half2float(halfArray[index]);
    }
}

__global__ void half2float_merge_k_iters(half *halfArray, float *floatArray, int N, int split_k_iters) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;

    if (index < N) {
        float sum = 0;
        for (int j = 0; j < split_k_iters; j++) {
            sum += __half2float(halfArray[index + j * N]);
        }
        floatArray[index] = sum;
    }
}

__global__ void merge_k_iters(half *input, half *output, int N, int split_k_iters) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;

    if (index < N) {
        half sum = 0;
        for (int j = 0; j < split_k_iters; j++) {
            sum = __hadd(sum, input[index + j * N]);
        }

        output[index] = sum;
    }
}

__global__ void merge_k_iters_qkv(half *input, half *output, int N, int split_k_iters) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;

    if (index < N) {
        half sum = 0;
        for (int j = 0; j < split_k_iters; j++) {
            sum = __hadd(sum, input[index + j * N / 3]);
        }

        output[index] = sum;
    }
}

// Explicitly instantiate the generic template function for other types (if needed)
template void allocate_aligned_memory_gpu(float*& ptr, size_t size);
template void allocate_aligned_memory_gpu(int*& ptr, size_t size);
template void allocate_aligned_memory_gpu(int8_t*& ptr, size_t size);
template void allocate_aligned_memory_gpu(uint8_t*& ptr, size_t size);
template void allocate_aligned_memory_gpu(half*& ptr, size_t size);
template void allocate_aligned_memory_gpu(half_float::half*& ptr, size_t size);

template void free_aligned_memory_gpu(float*& ptr);
template void free_aligned_memory_gpu(int*& ptr);
template void free_aligned_memory_gpu(int8_t*& ptr);
template void free_aligned_memory_gpu(uint8_t*& ptr);
template void free_aligned_memory_gpu(half*& ptr);
template void free_aligned_memory_gpu(half_float::half*& ptr);
