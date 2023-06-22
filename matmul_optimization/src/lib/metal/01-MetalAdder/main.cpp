// Lars Gebraad, 20th of April, 2022
//

#include <iostream>
#include <omp.h>

#define NS_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#include "Foundation/Foundation.hpp"
#include "Metal/Metal.hpp"
#include "QuartzCore/QuartzCore.hpp"

#include "MetalAdder.hpp"

template <class T>
void add_array_openmp(const T *a, const T *b, T *c, size_t length);
template <class T>
void add_array_serial(const T *a, const T *b, T *c, size_t length);
int omp_thread_count();
template <class T>
void statistics(T *array, size_t length, T &array_mean, T &array_std);

typedef std::chrono::microseconds time_unit;
auto unit_name = "microseconds";

int main(int argc, char *argv[])
{
    // Create GPU code / arrays --------------------------------------------------------
    MTL::Device *device = MTL::CreateSystemDefaultDevice();
    MetalAdder *adder = new MetalAdder(device);

    // Verify Metal code ---------------------------------------------------------------
    adder->sendComputeCommand(); // This computes the array sum
    adder->verifyResults();

    // Profile Metal code --------------------------------------------------------------
    int repeats = 100;
    auto durations = new float[repeats];
    for (size_t repeat = 0; repeat < repeats; repeat++)
    {
        auto start = std::chrono::high_resolution_clock::now();
        adder->sendComputeCommand();
        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<time_unit>(stop - start).count();
        durations[repeat] = duration;
    }
    float array_mean;
    float array_std;
    statistics(durations, repeats, array_mean, array_std);
    std::cout << "Metal (GPU) code performance: " << std::endl;
    std::cout << array_mean << unit_name << " \t +/- " << array_std << unit_name << std::endl
              << std::endl;

    // Verify serial code --------------------------------------------------------------
    // Get buffers pointers for CPU code. Using MTL::ResourceStorageModeShared should
    // make them accessible to both GPU and CPU, perfect!
    auto array_a = ((float *)adder->_mBufferA->contents());
    auto array_b = ((float *)adder->_mBufferB->contents());
    auto array_c = ((float *)adder->_mBufferResult->contents());

    // Let's randomize the data again, making sure that the result buffer starts out
    // incorrect
    adder->prepareData();
    add_array_serial(array_a, array_b, array_c, arrayLength);
    adder->verifyResults();

    // Profile serial code -------------------------------------------------------------
    for (size_t repeat = 0; repeat < repeats; repeat++)
    {
        auto start = std::chrono::high_resolution_clock::now();
        add_array_serial(array_a, array_b, array_c, arrayLength);
        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<time_unit>(stop - start).count();
        durations[repeat] = duration;
    }
    statistics(durations, repeats, array_mean, array_std);
    std::cout << "Serial code performance: " << std::endl;
    std::cout << array_mean << unit_name << " \t +/- " << array_std << unit_name << std::endl
              << std::endl;

    size_t max_threads = 10;
    for (size_t threads = 1; threads <= max_threads; threads++)
    {

        // Verify OpenMP code --------------------------------------------------------------
        omp_set_num_threads(threads);
        adder->prepareData();
        add_array_openmp(array_a, array_b, array_c, arrayLength);
        adder->verifyResults();

        // Profile OpenMP code -------------------------------------------------------------
        for (size_t repeat = 0; repeat < repeats; repeat++)
        {
            auto start = std::chrono::high_resolution_clock::now();
            add_array_openmp(array_a, array_b, array_c, arrayLength);
            auto stop = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<time_unit>(stop - start).count();
            durations[repeat] = duration;
        }
        statistics(durations, repeats, array_mean, array_std);
        std::cout << "OpenMP (" << omp_thread_count() << " threads) code performance: " << std::endl;
        std::cout << array_mean << unit_name << " \t +/- " << array_std << unit_name << std::endl
                  << std::endl;
    }

    delete[] durations;
    delete adder;
    device->release();
}

template <class T>
void add_array_openmp(const T *a, const T *b, T *c, size_t length)
{
    // Compute array sum a+b=c parallely using OpenMP, template function
#pragma omp parallel for
    for (
        size_t i = 0; i < length; i++)
    {
        c[i] = a[i] + b[i];
    }
}

template <class T>
void add_array_serial(const T *a, const T *b, T *c, size_t length)
{
    // Compute array sum a+b=c serially, template function
    for (size_t i = 0; i < length; i++)
    {
        c[i] = a[i] + b[i];
    }
}

int omp_thread_count()
{
    int n = 0;
#pragma omp parallel reduction(+ \
                               : n)
    n += 1;
    return n;
}

template <class T>
void statistics(T *array, size_t length, T &array_mean, T &array_std)
{
    // Compute mean and standard deviation serially, template function

    array_mean = 0;
    for (size_t repeat = 0; repeat < length; repeat++)
    {
        array_mean += array[repeat];
    }
    array_mean /= length;

    array_std = 0;
    for (size_t repeat = 0; repeat < length; repeat++)
    {
        array_std += pow(array_mean - array[repeat], 2.0);
    }
    array_std /= length;
    array_std = pow(array_std, 0.5);
}