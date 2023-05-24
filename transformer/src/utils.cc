#include "utils.h"

#include <stdio.h>

#include <cassert>
#include <cerrno>  // for errno
#include <cmath>
#include <cstdlib>
#include <cstring>  // for strerror
#include <iostream>

template <typename T>
void read_to_array(const char* path, T* array, int size) {
    std::ifstream infile(path, std::ios::binary | std::ios::in);
    if (infile.fail()) {
        std::cout << "Failed to open file: " << strerror(errno) << std::endl;
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

template <typename T>
void allocate_aligned_memory(T*& ptr, size_t size) {
    constexpr size_t alignment = 32;

    int ret = posix_memalign((void**)(&ptr), alignment, size);

    if (ret != 0) {
        throw("Memory allocation failed.");
    }
}

// Explicitly instantiate the generic template function for other types (if needed)
template bool check_two_equal<float>(float* array, float* array2, int size);
template void read_to_array<float>(const char* path, float* array, int size);
template void read_to_array<int8_t>(const char* path, int8_t* array, int size);
template void read_to_array<int32_t>(const char* path, int32_t* array, int size);
template void allocate_aligned_memory(float*& ptr, size_t size);
template void allocate_aligned_memory(int8_t*& ptr, size_t size);
template void allocate_aligned_memory(int*& ptr, size_t size);
