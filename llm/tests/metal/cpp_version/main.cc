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
#include "param.h"

// .h
MTL::Buffer *bM1, *bM2, *bM3, *bParam;
MTL::Device* mDevice;
MTL::ComputePipelineState* mfnPipelineState;
MTL::CommandQueue* mCommandQueue;
NS::Error *error = nullptr;

using namespace std;
using namespace chrono;

// .cc
const char * fn_name = "matmul";

// main
uint height1 = 100;
uint width1 = 100;
uint height2 = 100;
uint width2 = 100;
float *A1, *A2, *A3;
matmul_param *param;

// Test Use
void test_addArrays(const float arr1[], const float arr2[], float result[], uint size) {
    for (int i = 0; i < size; ++i) {
        result[i] = arr1[i] + arr2[i];
    }
}
void test_matmul(const float* matA, int rowsA, int colsA, const float* matB, int rowsB, int colsB, float* result) {
    for (int i = 0; i < rowsA; i++) {
        for (int j = 0; j < colsB; j++) {
            result[i * colsB + j] = 0;
            for (int k = 0; k < colsA; k++) {
                result[i * colsB + j] += matA[i * colsA + k] * matB[k * colsB + j];
            }
        }
    }
}
void printArray(const float* array, uint arraySize) {
    for (int i = 0; i < arraySize; ++i) {
        std::cout << array[i] << " ";
    }
    std::cout << std::endl;
}
void generateRandomFloatArray(float* array, uint arraySize) {
    // Use a random device to seed the random number generator
    std::random_device rd;
    // Use the current time as a seed for the random number generator
    std::mt19937 gen(rd());
    // Define the range of random numbers (adjust as needed)
    std::uniform_real_distribution<float> distribution(1, 100);

    // Generate random integers and fill the array
    for (int i = 0; i < arraySize; ++i) {
        array[i] = distribution(gen);
    }
}

// Metal functions
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
    bM1 = metal_newBuf(sizeof(float), param->arraySize1);
    bM2 = metal_newBuf(sizeof(float), param->arraySize2);
    bM3 = metal_newBuf(sizeof(float), param->outputsize);
    bParam = metal_newBuf(sizeof(matmul_param), 1);

    computeEncoder->setComputePipelineState(mfnPipelineState);
    computeEncoder->setBuffer(bM1, 0, 0);
    computeEncoder->setBuffer(bM2, 0, 1);
    computeEncoder->setBuffer(bParam, 0, 2);
    computeEncoder->setBuffer(bM3, 0, 3);
    

    memcpy(bM1->contents(), A1, param->arraySize1*sizeof(float));
    memcpy(bM2->contents(), A2, param->arraySize2*sizeof(float));
    memcpy(bM3->contents(), A3, param->outputsize*sizeof(float));
    memcpy(bParam->contents(), param, sizeof(matmul_param));
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
    MTL::Size mThreadGroupSize = MTL::Size::Make(8, 8, 1);
    MTL::Size mGridSize = MTL::Size::Make((param->width1 + mThreadGroupSize.width - 1) / mThreadGroupSize.width,
                                           (param->height2 + mThreadGroupSize.height - 1) / mThreadGroupSize.height,
                                           1);

    // Dispatch and Run Computation
    // auto start = high_resolution_clock::now();
    computeEncoder->dispatchThreadgroups(mGridSize, mThreadGroupSize);
    computeEncoder->endEncoding();
    commandBuffer->commit();
    commandBuffer->waitUntilCompleted();
    // auto stop = high_resolution_clock::now();
    // auto duration = duration_cast<microseconds>(stop - start);
    // std::cout << "GPU: " << duration.count() << "ms" << std::endl;
    computeEncoder->release();
    commandBuffer->release();
}

int main(){
    // Initialization for array addition
    param = new matmul_param;
    param->height1 = height1;
    param->height2 = height2;
    param->width1 = width1;
    param->width2 = width2;
    param->outputsize = height1*width2;
    param->arraySize1 = width1*height1;
    param->arraySize2 = width2*height2;
    A1 = new float[param->arraySize1];
    A2 = new float[param->arraySize2];
    A3 = new float[param->outputsize];
    generateRandomFloatArray(A1, param->arraySize1);
    generateRandomFloatArray(A2, param->arraySize2);
    // printArray(A1, param->arraySize1);
    // printArray(A2, param->arraySize2);
    

    // CPU
    // auto start = high_resolution_clock::now();
    // test_matmul(A1, param->height1,  param->width1, A2, param->height2, param->width2, A3);
    // auto stop = high_resolution_clock::now();
    // auto duration = duration_cast<microseconds>(stop - start);
    // std::cout << "CPU: " << duration.count() << "ms" << std::endl;
    // printf("CPU Results: \n");
    // for (uint32_t i = 0; i < param->outputsize; i++){
    //     printf("A3[%d]: %f\n", i, A3[i]);
    // }
    // free(A3);
    // A3 = new float[param->outputsize];
    
    // GPU
    metal_init();
    metal_compute();
    // printf("GPU Results: \n");
    // for (uint32_t i = 0; i < param->outputsize; i++){
    //     printf("bM3[%d]: %f\n", i, ((float*)(bM3->contents()))[i]);
    // }

    free(A1);
    free(A2);
    free(A3);
}




