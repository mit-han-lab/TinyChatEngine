//
//  main.cpp
//  metal_cpp
//
//  Created by Derrick on 1/24/24.
//  Some to-do list:
//  1. keep a map: ptr on CPU -> buffer on GPU

#include <iostream>
#include <random>
#include <ctime>
#include <chrono>

#define NS_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION

#include "Metal/Metal.hpp"
#include "Foundation/Foundation.hpp"

MTL::Buffer *bM1, *bM2, *bM3;
MTL::Device* mDevice;
MTL::ComputePipelineState* mfnPipelineState;
MTL::CommandQueue* mCommandQueue;
NS::Error *error = nullptr;

const char * fn_name = "arrayAdd";

int *A1, *A2, *A3;


using namespace std;
using namespace chrono;

uint row = 100;
uint col = 100;
uint arraySize = row*col;

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

void metal_init(){
    mDevice = MTL::CreateSystemDefaultDevice();
    MTL::Library *defaultLibrary = mDevice->newDefaultLibrary();
    if (defaultLibrary == nullptr) {
        std::cout << "Failed to find the default library." << std::endl;
        return;
    }
    auto str = NS::String::string(fn_name, NS::ASCIIStringEncoding);
    MTL::Function *matmulFunction = defaultLibrary->newFunction(str);
    defaultLibrary->release();
    if (matmulFunction == nullptr) {
        std::cout << "Failed to find the function." << std::endl;
        return;
    }
    mfnPipelineState = mDevice->newComputePipelineState(matmulFunction, &error);
    matmulFunction->release();
    if (mfnPipelineState == nullptr) {
        std::cout << "Failed to created pipeline state object, error " << error << "." << std::endl;
        return;
    }
    mCommandQueue = mDevice->newCommandQueue();
    if (mCommandQueue == nullptr) {
        std::cout << "Failed to find the command queue." << std::endl;
        return;
    }
}

MTL::Buffer *metal_newBuf(unsigned long type_size, unsigned long size){
    return mDevice->newBuffer(type_size*size, MTL::ResourceStorageModeShared);
}

void metal_encodecommand(MTL::ComputeCommandEncoder *computeEncoder){
    //Create Metal buffers for input and output, if inside the TinyChat, param should be created in advance
    bM1 = metal_newBuf(sizeof(int), arraySize);
    bM2 = metal_newBuf(sizeof(int), arraySize);
    bM3 = metal_newBuf(sizeof(int), arraySize);

    computeEncoder->setComputePipelineState(mfnPipelineState);
    computeEncoder->setBuffer(bM1, 0, 0);
    computeEncoder->setBuffer(bM2, 0, 1);
    computeEncoder->setBuffer(bM3, 0, 2);

    memcpy(bM1->contents(), A1, arraySize);
    memcpy(bM2->contents(), A2, arraySize);
    memcpy(bM3->contents(), A3, arraySize);
}

void metal_compute(){
    // Initialization of GPU vals
    MTL::CommandBuffer *commandBuffer = mCommandQueue->commandBuffer();
    assert(commandBuffer != nullptr);
    MTL::ComputeCommandEncoder *computeEncoder = commandBuffer->computeCommandEncoder();
    assert(computeEncoder != nullptr);
    
    // Encode command and set buffer to GPU
    metal_encodecommand(computeEncoder);

    // Threads -> ThreadGroup -> Grid
    NS::UInteger maxThreadGroupSize = mfnPipelineState->maxTotalThreadsPerThreadgroup();
    NS::UInteger ThreadGroupSize = MIN(arraySize, maxThreadGroupSize);
    MTL::Size mGridSize = MTL::Size::Make((arraySize + ThreadGroupSize - 1) / ThreadGroupSize, 1, 1);
    MTL::Size mThreadGroupSize = MTL::Size::Make(ThreadGroupSize, 1, 1);

    // Dispatch and Run Computation
    computeEncoder->dispatchThreadgroups(mGridSize, mThreadGroupSize);
    computeEncoder->endEncoding();
    commandBuffer->commit();
    commandBuffer->waitUntilCompleted();
    computeEncoder->release();
    commandBuffer->release();
}

int main(){

    // Initialization for array addition
    A1 = new int[arraySize];
    A2 = new int[arraySize];
    A3 = new int[arraySize];
    generateRandomIntArray(A1);
    generateRandomIntArray(A2);

    // Initialization for matmul
    
    metal_init();
    metal_compute();
    printf("A1: %d; A2 %d; A3 %d\n", A1[0], A2[0], ((int*)(bM3->contents()))[0]);
    
    
}   




