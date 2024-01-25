//
//  main.cpp
//  metal_cpp
//
//  Created by Derrick on 1/24/24.
//

#include <iostream>
#include <random>
#include <ctime>

#define NS_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#include "Metal/Metal.hpp"
#include "Foundation/Foundation.hpp"

int arraySize = 100;

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
    int *Output = new int[arraySize];
    
    generateRandomIntArray(M1);
    generateRandomIntArray(M2);
    generateRandomIntArray(Output);
    
    
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
    MTL::Buffer *buffer1 = _mDevice->newBuffer(sizeof(M1), MTL::ResourceStorageModeShared);
    MTL::Buffer *buffer2 = _mDevice->newBuffer(sizeof(M2), MTL::ResourceStorageModeShared);
    MTL::Buffer *buffer3 = _mDevice->newBuffer(sizeof(Output), MTL::ResourceStorageModeShared);
    
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
    computeEncoder->setBuffer(buffer1, 0, 0);
    computeEncoder->setBuffer(buffer2, 0, 1);
    computeEncoder->setBuffer(buffer3, 0, 2);
    
    // number of threadgroup
    uint32_t maxThreadsperthreadgroup = (uint32_t)_mMatmulFunctionPSO->maxTotalThreadsPerThreadgroup();
    uint32_t threadsPerthreadgroup = MIN(maxThreadsperthreadgroup, arraySize);
    MTL::Size threadgroupCount = MTL::Size::Make((arraySize+threadsPerthreadgroup-1)/threadsPerthreadgroup, 1, 1);

    // Calculate a thread number per group
    MTL::Size threadgroupSize = MTL::Size::Make(threadsPerthreadgroup, 1, 1);

    // Encode the compute command.
    computeEncoder->dispatchThreads(threadgroupCount, threadgroupSize);

    // End the compute pass.
    computeEncoder->endEncoding();

    // Execute the command.
    commandBuffer->commit();

    // Normally, you want to do other work in your app while the GPU is running,
    // but in this example, the code simply blocks until the calculation is complete.
    commandBuffer->waitUntilCompleted();
    
    int *output = (int*)buffer3->contents();
    std::cout << "The output from Metal GPU is: " << output[0] << std::endl;

    computeEncoder->release();
    commandBuffer->release();
    
}

