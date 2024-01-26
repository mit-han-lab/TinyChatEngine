//
//  main.cpp
//  metal_cpp
//
//  Created by Derrick on 1/24/24.
//

#include <iostream>
#include <random>
#include <ctime>
#include <chrono>

#define NS_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#include "Metal/Metal.hpp"
#include "Foundation/Foundation.hpp"

MTL::Buffer* pM1;
MTL::Buffer* pM2;
MTL::Buffer* pM3;

using namespace std;
using namespace chrono;

int arraySize = 100000000;

void addArrays(const int arr1[], const int arr2[], int result[], int size) {
    for (int i = 0; i < size; ++i) {
        result[i] = arr1[i] + arr2[i];
    }
}

// Function to generate a random integer array
void generateRandomIntArray(int* array) {
    // Use a random device to seed the random number generator
    std::random_device rd;
    // Use the current time as a seed for the random number generator
    std::mt19937 gen(rd());
    // Define the range of random numbers (adjust as needed)
    std::uniform_int_distribution<int> distribution(1, 100);

    // Generate random integers and fill the array
    for (int i = 0; i < arraySize; ++i) {
        array[i] = distribution(gen);
    }
}

int main(){
//    int M1[5][5], M2[5][5], Output[5][5];
    int *M1 = new int[arraySize];
    int *M2 = new int[arraySize];
    int *M3 = new int[arraySize];

    generateRandomIntArray(M1);
    generateRandomIntArray(M2);
    generateRandomIntArray(M3);

    auto start2 = high_resolution_clock::now();
    addArrays(M1, M2, M3, arraySize);
    auto stop2 = high_resolution_clock::now();
    auto duration2 = duration_cast<microseconds>(stop2 - start2);
    std::cout << "CPU" << std::endl;
    std::cout << "M1[0]: " << M1[0] << " " << M1[1] << " " << M1[2] << std::endl;
    std::cout << "M2[0]: " << M2[0] << " " << M2[1] << " " << M2[2] << std::endl;
    std::cout << "M3[0]: " << M3[0] << " " << M3[1] << " " << M3[2]  << std::endl;
    
    // auto start = high_resolution_clock::now();
    MTL::Device *_mDevice = MTL::CreateSystemDefaultDevice();
    NS::Error *error = nullptr;
    MTL::Library *defaultLibrary = _mDevice->newDefaultLibrary();
    
    if (defaultLibrary == nullptr) {
        std::cout << "Failed to find the default library." << std::endl;
        return 0;
    }
    
    // Give matmul kernel
    auto str = NS::String::string("arrayAdd", NS::ASCIIStringEncoding);
    MTL::Function *matmulFunction = defaultLibrary->newFunction(str);
    defaultLibrary->release();
    
    if (matmulFunction == nullptr) {
        std::cout << "Failed to find the function." << std::endl;
        return 0;
    }
    
    // Create a compute pipeline state object.
    MTL::ComputePipelineState * _mMatmulFunctionPSO = _mDevice->newComputePipelineState(matmulFunction, &error);
    matmulFunction->release();
    
    if (_mMatmulFunctionPSO == nullptr) {
        //  If the Metal API validation is enabled, you can find out more information about what
        //  went wrong.  (Metal API validation is enabled by default when a debug build is run
        //  from Xcode)
        std::cout << "Failed to created pipeline state object, error " << error << "." << std::endl;
        return 0;
    }
    
    MTL::CommandQueue * _mCommandQueue = _mDevice->newCommandQueue();
    if (_mCommandQueue == nullptr) {
        std::cout << "Failed to find the command queue." << std::endl;
        return 0;
    }
    
    //Create Metal buffers for input and output, if inside the TinyChat, param should be created in advance
    MTL::Buffer *buffer1 = _mDevice->newBuffer(sizeof(int)*arraySize, MTL::ResourceStorageModeShared);
    MTL::Buffer *buffer2 = _mDevice->newBuffer(sizeof(int)*arraySize, MTL::ResourceStorageModeShared);
    MTL::Buffer *buffer3 = _mDevice->newBuffer(sizeof(int)*arraySize, MTL::ResourceStorageModeShared);
    
    pM1 = buffer1;
    pM2 = buffer2;
    pM3 = buffer3;
    memcpy(pM1->contents(), M1, arraySize);
    memcpy(pM2->contents(), M2, arraySize);
    memcpy(pM3->contents(), M3, arraySize);
    
    // Start the computation in metal gpu
    // Create a command buffer to hold commands.
    MTL::CommandBuffer *commandBuffer = _mCommandQueue->commandBuffer();
    assert(commandBuffer != nullptr);

    // Start a compute pass.
    MTL::ComputeCommandEncoder *computeEncoder = commandBuffer->computeCommandEncoder();
    assert(computeEncoder != nullptr);

    // Set buffers for input and output
    // Encode the pipeline state object and its parameters.
    computeEncoder->setComputePipelineState(_mMatmulFunctionPSO);
    computeEncoder->setBuffer(pM1, 0, 0);
    computeEncoder->setBuffer(pM2, 0, 1);
    computeEncoder->setBuffer(pM3, 0, 2);
    
    // number of threadgroup
    MTL::Size mtlthreadsPerthreadgroup = MTL::Size::Make(arraySize, 1, 1);
    // Calculate a thread number per group
    MTL::Size threadgroupSize = MTL::Size::Make(1, 1, 1);


    // Set threadgroup size and dispatch compute threads
    NS::UInteger maxThreadsperthreadgroup = _mMatmulFunctionPSO->maxTotalThreadsPerThreadgroup();
    NS::UInteger threadsPerThreadgroup = MIN(arraySize, maxThreadsperthreadgroup);
    MTL::Size threadgroupCount = MTL::Size::Make((arraySize + threadsPerThreadgroup - 1) / threadsPerThreadgroup, 1, 1);
    // Dispatch threads in multiple threadgroups
    MTL::Size threadgroups = MTL::Size::Make(threadsPerThreadgroup, 1, 1);

    auto start = high_resolution_clock::now();
    // Encode the compute command.
    // computeEncoder->dispatchThreads(mtlthreadsPerthreadgroup, threadgroupSize);
    computeEncoder->dispatchThreadgroups(threadgroups, threadgroupCount);

    // End the compute pass.
    computeEncoder->endEncoding();

    // Execute the command.
    commandBuffer->commit();

    // Normally, you want to do other work in your app while the GPU is running,
    // but in this example, the code simply blocks until the calculation is complete.
    commandBuffer->waitUntilCompleted();
    
    std::cout << "GPU" << std::endl;
    std::cout << "M1[0]: " << ((int*)(buffer1->contents()))[0] << " " << ((int*)(buffer1->contents()))[1] << " " << ((int*)(buffer1->contents()))[2] << std::endl;
    std::cout << "M2[0]: " << ((int*)(buffer2->contents()))[0] << " " << ((int*)(buffer2->contents()))[1] << " " << ((int*)(buffer2->contents()))[2] << std::endl;
    std::cout << "M3[0]: " << ((int*)(buffer3->contents()))[0] << " " << ((int*)(buffer3->contents()))[1] << " " << ((int*)(buffer3->contents()))[2]  << std::endl;
    
    computeEncoder->release();
    commandBuffer->release();
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);

    cout << "GPU: " << duration.count() << " microseconds" << endl;
    cout << "CPU: " << duration2.count() << " microseconds" << endl;
}   




