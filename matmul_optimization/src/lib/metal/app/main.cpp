// Lars Gebraad, 20th of April, 2022
//

#include <omp.h>

#include <iostream>

#define NS_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#include "Foundation/Foundation.hpp"
#include "Metal/Metal.hpp"
#include "MetalMatmul.hpp"
#include "MetalMatmulInt4.hpp"
#include "QuartzCore/QuartzCore.hpp"

template <class T>
void add_array_openmp(const T *a, const T *b, T *c, size_t length);
template <class T>
void add_array_serial(const T *a, const T *b, T *c, size_t length);
int omp_thread_count();
template <class T>
void statistics(T *array, size_t length, T &array_mean, T &array_std);

typedef std::chrono::microseconds time_unit;
auto unit_name = "microseconds";

int main(int argc, char *argv[]) {
    MetalMatMulParams params = {1, 4096, 32000, 32};
    // Create GPU code / arrays --------------------------------------------------------
    MTL::Device *device = MTL::CreateSystemDefaultDevice();
    MetalMatmulInt4 *op = new MetalMatmulInt4(device, params);
    MetalMatmulInt4 *op2 = new MetalMatmulInt4(device, params);
    MetalMatmulInt4 *op3 = new MetalMatmulInt4(device, params);

    // Verify Metal code ---------------------------------------------------------------
    op->sendComputeCommand();  // This computes the array sum
    op->verifyResults();

    // warm up
    int warm_up = 10;
    for (size_t repeat = 0; repeat < warm_up; repeat++) {
        op->sendComputeCommand();
    }

    // mimic interleaving with other ops
    op2->sendComputeCommand();
    op3->sendComputeCommand();

    // Profile Metal code --------------------------------------------------------------
    int repeats = 1;
    auto durations = new float[repeats];
    for (size_t repeat = 0; repeat < repeats; repeat++) {
        auto start = std::chrono::high_resolution_clock::now();
        op->sendComputeCommand();
        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<time_unit>(stop - start).count();
        durations[repeat] = duration;
    }
    float array_mean;
    float array_std;
    statistics(durations, repeats, array_mean, array_std);
    std::cout << "Metal (GPU) code performance: " << std::endl;
    std::cout << array_mean << unit_name << " \t +/- " << array_std << unit_name << std::endl << std::endl;

    // Verify serial code --------------------------------------------------------------
    // Get buffers pointers for CPU code. Using MTL::ResourceStorageModeShared should
    // make them accessible to both GPU and CPU, perfect!
    auto array_a = ((float *)op->_mBufferA->contents());
    auto array_b = ((float *)op->_mBufferB->contents());
    auto array_c = ((float *)op->_mBufferResult->contents());

    delete[] durations;
    delete op;
    device->release();
}

template <class T>
void add_array_openmp(const T *a, const T *b, T *c, size_t length) {
    // Compute array sum a+b=c parallely using OpenMP, template function
#pragma omp parallel for
    for (size_t i = 0; i < length; i++) {
        c[i] = a[i] + b[i];
    }
}

template <class T>
void add_array_serial(const T *a, const T *b, T *c, size_t length) {
    // Compute array sum a+b=c serially, template function
    for (size_t i = 0; i < length; i++) {
        c[i] = a[i] + b[i];
    }
}

int omp_thread_count() {
    int n = 0;
#pragma omp parallel reduction(+ : n)
    n += 1;
    return n;
}

template <class T>
void statistics(T *array, size_t length, T &array_mean, T &array_std) {
    // Compute mean and standard deviation serially, template function

    array_mean = 0;
    for (size_t repeat = 0; repeat < length; repeat++) {
        array_mean += array[repeat];
    }
    array_mean /= length;

    array_std = 0;
    for (size_t repeat = 0; repeat < length; repeat++) {
        array_std += pow(array_mean - array[repeat], 2.0);
    }
    array_std /= length;
    array_std = pow(array_std, 0.5);
}
