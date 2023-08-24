#include "utils.h"

#include <stdio.h>

#include <cassert>
#include <cerrno>  // for errno
#include <cmath>
#include <cstdlib>
#include <cstring>  // for strerror
#include <iostream>

#include "common.h"

// To be deprecated soon
template <typename T>
void read_to_array(const char* path, T* array, int size) {
    std::ifstream infile(path, std::ios::binary | std::ios::in);
    if (infile.fail()) {
        std::cout << strerror(errno) << ": " << path << std::endl;
        throw("Expected error...");
    } else {
        infile.read(reinterpret_cast<char*>(array), size * sizeof(T));
        infile.close();
    }
}

struct max_error_info {
    int idx;
    float a1, a2;
};

bool check_two_equal(float* array, float* array2, int size, float error) {
    float sq_diff = 0;
    float max_sqdiff = 0;
    struct max_error_info error_info;

    for (int i = 0; i < size; i++) {
        float diff = (float)array[i] - (float)array2[i];
        if (std::isnan(diff)) return false;
        sq_diff += diff * diff;
        if (diff * diff > max_sqdiff) {
            max_sqdiff = diff * diff;
            error_info.idx = i;
            error_info.a1 = array[i];
            error_info.a2 = array2[i];
        }
    }
    if ((sq_diff / size) > error) {
        std::cout << "MSE:" << sq_diff / size << ", MAX SQ diff:" << max_sqdiff;
        std::cout << "@:" << error_info.idx << ",a1:" << error_info.a1 << ",a2:" << error_info.a2 << std::endl;
        return false;
    }
    return true;
}

template <typename T>
bool check_two_equal(T* array, T* array2, int size) {
    float sq_diff = 0;
    float max_sqdiff = 0;
    for (int i = 0; i < size; i++) {
        float diff = (float)array[i] - (float)array2[i];
        sq_diff += diff * diff;
        if (diff * diff > max_sqdiff) max_sqdiff = diff * diff;
        if (sqrt(max_sqdiff) > MAX_SQ_ERROR_MAX) {
            std::cout << "i:" << i << ",max_sqdiff:" << sqrt(max_sqdiff) << ", array[i]:";
            std::cout << static_cast<float>(array[i]) << ", array2[i]:" << static_cast<float>(array2[i]) << std::endl;
            return false;
        }
    }
    if ((sq_diff / size) > ERROR_MAX) {
        std::cout << "MSE:" << sq_diff / size << ", MAX SQ diff:" << max_sqdiff << std::endl;
        return false;
    }
    return true;
}

template <>
bool check_two_equal<int8_t>(int8_t* array, int8_t* array2, int size) {
    float sq_diff = 0;
    float max_sqdiff = 0;
    for (int i = 0; i < size; i++) {
        float diff = (float)array[i] - (float)array2[i];
        sq_diff += diff * diff;
        if (diff * diff > max_sqdiff) max_sqdiff = diff * diff;
    }
    if ((sq_diff / size) > INT_ERROR_MAX) {
        std::cout << "MSE:" << sq_diff / size << ", MAX SQ diff:" << max_sqdiff << std::endl;
        return false;
    }
    return true;
}

bool check_two_exact_equal(int8_t* array, int8_t* array2, int size) {
    float sq_diff = 0;
    float max_sqdiff = 0;
    for (int i = 0; i < size; i++) {
        if (array[i] != array2[i]) {
            std::cout << "i:" << i << ", array[i]:" << static_cast<int>(array[i])
                      << ", array2[i]:" << static_cast<int>(array2[i]) << std::endl;
            return false;
        }
    }
    return true;
}

bool check_two_equal(int8_t* array, int8_t* array2, int size, float error) {
    float sq_diff = 0;
    float max_sqdiff = 0;
    for (int i = 0; i < size; i++) {
        // if (array[i] != array2[i]) {
        // std::cout << "i:" << i << ", array[i]:" << static_cast<int>(array[i]) << ", array2[i]:" <<
        // static_cast<int>(array2[i]) << std::endl;
        // return false;
        // }
        float diff = (float)array[i] - (float)array2[i];
        sq_diff += diff * diff;
        if (diff * diff > max_sqdiff) max_sqdiff = diff * diff;
    }
    if ((sq_diff / size) > error) {
        std::cout << "MSE:" << sq_diff / size << ", MAX SQ diff:" << max_sqdiff << std::endl;
        return false;
    }
    return true;
}

void print_MSE_max_diff(float* a, float* a2, int size) {
    float sq_diff = 0;
    float max_sqdiff = 0;
    struct max_error_info error_info;

    for (int i = 0; i < size; i++) {
        float diff = (float)a[i] - (float)a2[i];
        sq_diff += diff * diff;
        if (diff * diff > max_sqdiff) {
            max_sqdiff = diff * diff;
            error_info.idx = i;
            error_info.a1 = a[i];
            error_info.a2 = a2[i];
        }
    }
    std::cout << "MSE:" << sq_diff / size << ", MAX SQ diff:" << max_sqdiff;
    std::cout << "@:" << error_info.idx << ",a1:" << error_info.a1 << ",a2:" << error_info.a2 << std::endl;
}

template <>
bool check_two_equal<int>(int* array, int* array2, int size) {
    float sq_diff = 0;
    for (int i = 0; i < size; i++) {
        float diff = (float)array[i] - (float)array2[i];
        sq_diff += diff * diff;
    }
    if ((sq_diff / size) > INT_ERROR_MAX) {
        std::cout << "MSE:" << sq_diff / size << std::endl;
        return false;
    }
    return true;
}

void print_first_k_elelment(std::string name, const int8_t* arr, int k, int start_idx) {
    std::cout << name << ":";
    for (int i = start_idx; i < k; i++) {
        std::cout << static_cast<int>(arr[i]) << ",";
    }
    std::cout << std::endl;
}

void print_first_k_elelment(std::string name, const int32_t* arr, int k, int start_idx) {
    std::cout << name << ":";
    for (int i = start_idx; i < k; i++) {
        std::cout << static_cast<int>(arr[i]) << ",";
    }
    std::cout << std::endl;
}

void print_first_k_elelment(std::string name, const float* arr, int k, int start_idx) {
    std::cout << name << ":";
    for (int i = start_idx; i < k; i++) {
        std::cout << static_cast<float>(arr[i]) << ",";
    }
    std::cout << std::endl;
}

#ifdef QM_METAL
// If we use metal GPU, let's make all the allocated memory accesible from GPU. We will
// 1. allocate the memory in to format of MTL::Buffer
// 2. make a mapping (unordered_map) between MTL::Buffer and memory address
// 3. when GPU want to access some address space, use the table to get the corresponding MTL::Buffer object
// Befenits: not to worry about memory alignment, better performance
#include "matmul_metal_int4_imp.h"

template <typename T>
void allocate_aligned_memory(T*& ptr, size_t size) {
    // allocate and get the pointer
    void* void_ptr = MetalMatmulInt4IMP::allocateSharedMem(size);
    if (void_ptr == NULL) {
        std::cerr << "Metal memory allocation failed." << std::endl;
        exit(-1);
    }
    ptr = (T*)void_ptr;
}

void deallocate_memory(void* ptr) { throw std::logic_error("Deallocate function not yet implemented"); }
#else
template <typename T>
void allocate_aligned_memory(T*& ptr, size_t size) {
    constexpr size_t alignment = 32;

#ifdef _WIN32
    // Windows version
    ptr = _aligned_malloc(size, alignment);
    int ret = (ptr != NULL) ? 0 : -1;
#else
    // POSIX compliant OS version
    int ret = posix_memalign((void**)(&ptr), alignment, size);
#endif

    if (ret != 0) {
        throw("Memory allocation failed.");
    }
}
void deallocate_memory(void* ptr) { free(ptr); }
#endif  // QM_METAL

// Explicitly instantiate the generic template function for other types (if needed)
template bool check_two_equal<float>(float* array, float* array2, int size);
template void read_to_array<float>(const char* path, float* array, int size);
template void read_to_array<int32_t>(const char* path, int32_t* array, int size);
template void read_to_array<int8_t>(const char* path, int8_t* array, int size);
template void read_to_array<uint8_t>(const char* path, uint8_t* array, int size);
template void allocate_aligned_memory(float*& ptr, size_t size);
template void allocate_aligned_memory(int*& ptr, size_t size);
template void allocate_aligned_memory(int8_t*& ptr, size_t size);
template void allocate_aligned_memory(uint8_t*& ptr, size_t size);
template void allocate_aligned_memory(pack_q4_tensor*& ptr, size_t size);
