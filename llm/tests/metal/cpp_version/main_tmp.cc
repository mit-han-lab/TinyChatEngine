//
//  main.cpp
//  metal_cpp
//
//  Created by Derrick on 1/24/24.
//  Some to-do list:
//  1. keep a map: ptr on CPU -> buffer on GPU
// Notes:
// 1. Offset hasn't been considered
// 2. Group_Size is multiple of 32

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
typedef struct{
    float values[4];
}PackedFloat;

typedef struct{
    unsigned char values[4];
}PackedChar;

// .h
MTL::Buffer *bM1, *bM2, *bM3, *bParam, *bScales, *bOffset;
MTL::Buffer *bsM1, *bsM2, *bsM3, *bsParam, *bsScales, *bsOffset;
MTL::Device* mDevice;
MTL::ComputePipelineState* mfnPipelineState;
MTL::CommandQueue* mCommandQueue;
NS::Error *error = nullptr;
typedef struct {
    float *A, *C, *scales, *offset;
    unsigned char *B;
} MetalMatmulBuffers;

typedef struct {
    float *C, *scales, *offset;
    PackedFloat *A;
    PackedChar *B;
} MetalMatmulSBuffers;

using namespace std;
using namespace chrono;

// .cc
const char * fn_name = "matmulInt4";


// main
unsigned int height1 = 32;
unsigned int width1 = 32;
unsigned int height2 = 32;
unsigned int width2 = 32;
float *A1, *A3;
unsigned char *A2;
matmul_param *param;
// for MatmulInt4 use
unsigned int group_size = 32;
float* scales, *offset;
MetalMatmulBuffers *Int4_buffer;
MetalMatMulParams *Int4_params;

MetalMatmulSBuffers *Int4_Sbuffer;
MetalMatMulParams *Int4_Sparams;

// Test Use
void test_addArrays(const float arr1[], const float arr2[], float result[], uint size) {
    for (int i = 0; i < size; ++i) {
        result[i] = arr1[i] + arr2[i];
    }
}
void test_matmul(const float* matA, int rowsA, int colsA, const unsigned char* matB, int rowsB, int colsB, float* result) {
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
void generateRandomCharArray(unsigned char* array, uint arraySize) {
    // Use a random device to seed the random number generator
    std::random_device rd;
    // Use the current time as a seed for the random number generator
    std::mt19937 gen(rd());
    // Define the range of random numbers (adjust as needed)
    std::uniform_int_distribution<unsigned char> distrib(0, 255);

    // Generate random integers and fill the array
    for (int i = 0; i < arraySize; ++i) {
        array[i] = static_cast<unsigned char>(distrib(gen));
    }
}

void generateRandomScale(float* array, uint arraySize) {
    for (size_t i = 0; i < arraySize; i++){
        array[i] = 1.1;
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

void metal_encodecommand_matmulInt4_simd(MTL::ComputeCommandEncoder *computeEncoder){
    //Create Metal buffers for input and output, if inside the TinyChat, param should be created in advance

    bsScales = metal_newBuf(sizeof(float), Int4_Sparams->height2*Int4_Sparams->width2/Int4_Sparams->group_size);
    bsM1 = metal_newBuf(sizeof(PackedFloat), Int4_Sparams->height1*Int4_Sparams->width1);
    bsM2 = metal_newBuf(sizeof(PackedChar), Int4_Sparams->width2*Int4_Sparams->width2);
    bsParam = metal_newBuf(sizeof(MetalMatMulParams), 1);
    bsM3 = metal_newBuf(sizeof(float), Int4_Sparams->height3*Int4_Sparams->width3);
    
    computeEncoder->setComputePipelineState(mfnPipelineState);
    computeEncoder->setBuffer(bsM1, 0, 0);
    computeEncoder->setBuffer(bsM2, 0, 1);
    computeEncoder->setBuffer(bsM3, 0, 2);
    computeEncoder->setBuffer(bsScales, 0, 3);
    computeEncoder->setBuffer(bParam, 0, 4);
    
    memcpy(bsM1->contents(), Int4_Sbuffer->A, Int4_Sparams->height1*Int4_Sparams->width1*sizeof(PackedFloat));
    memcpy(bsM2->contents(), Int4_Sbuffer->B, Int4_Sparams->width2*Int4_Sparams->width2*sizeof(PackedChar));
    memcpy(bsM3->contents(), Int4_Sbuffer->C, Int4_Sparams->height3*Int4_Sparams->width3*sizeof(float));
    memcpy(bsParam->contents(), Int4_Sparams, sizeof(MetalMatMulParams));
    memcpy(bsScales->contents(), Int4_Sbuffer->scales, Int4_Sparams->height2*Int4_Sparams->width2/Int4_Sparams->group_size*sizeof(float));
}


void metal_encodecommand_matmulInt4(MTL::ComputeCommandEncoder *computeEncoder){
    //Create Metal buffers for input and output, if inside the TinyChat, param should be created in advance

    bScales = metal_newBuf(sizeof(float), height1*width1);
    bM1 = metal_newBuf(sizeof(float), Int4_params->height1*Int4_params->width1);
    bM2 = metal_newBuf(sizeof(unsigned char), Int4_params->width1*Int4_params->width3);
    bParam = metal_newBuf(sizeof(MetalMatMulParams), 1);
    bM3 = metal_newBuf(sizeof(float), Int4_params->height1*Int4_params->width3);
    
    computeEncoder->setComputePipelineState(mfnPipelineState);
    computeEncoder->setBuffer(bM1, 0, 0);
    computeEncoder->setBuffer(bM2, 0, 1);
    computeEncoder->setBuffer(bM3, 0, 2);
    computeEncoder->setBuffer(bScales, 0, 3);
    computeEncoder->setBuffer(bParam, 0, 4);
    
    memcpy(bM1->contents(), Int4_buffer->A, Int4_params->height1*Int4_params->width1*sizeof(float));
    memcpy(bM2->contents(), Int4_buffer->B, Int4_params->width1*Int4_params->width3*sizeof(unsigned char));
    memcpy(bM3->contents(), Int4_buffer->C, Int4_params->height1*Int4_params->width3*sizeof(float));
    memcpy(bParam->contents(), Int4_params, sizeof(MetalMatMulParams));
    memcpy(bScales->contents(), Int4_buffer->scales, height1*width1*sizeof(float));
}

void metal_encodecommand_matmul(MTL::ComputeCommandEncoder *computeEncoder){
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
    if (strcmp(fn_name, "matmulInt4") == 0) {
        metal_encodecommand_matmulInt4(computeEncoder);
    } else if (strcmp(fn_name, "matmul") == 0) {
        metal_encodecommand_matmul(computeEncoder);
    } else if (strcmp(fn_name, "matmulInt4_SIMD_Q4Interleave") == 0) {
        metal_encodecommand_matmulInt4_simd(computeEncoder);
    }
    
    // Threads -> ThreadGroup -> Grid
    MTL::Size mThreadGroupSize;
    MTL::Size mGridSize;
    if (strcmp(fn_name, "matmulInt4") == 0){
        mThreadGroupSize = MTL::Size::Make(Int4_params->width3, Int4_params->height1, 1);
        mGridSize = MTL::Size::Make(16, 1, 1);
    } else if (strcmp(fn_name, "matmul") == 0) {
        mThreadGroupSize = MTL::Size::Make(8, 8, 1);
        mGridSize = MTL::Size::Make((param->width1 + mThreadGroupSize.width - 1) / mThreadGroupSize.width,
                                              (param->height2 + mThreadGroupSize.height - 1) / mThreadGroupSize.height,
                                              1);
    } else if (strcmp(fn_name, "matmulInt4_SIMD_Q4Interleave") == 0) {
        mThreadGroupSize = MTL::Size::Make(Int4_Sparams->width3, Int4_Sparams->height1, 1);
        mGridSize = MTL::Size::Make(16, 1, 1);
    }
    
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

void test_normal_matmul(){
    // Initialization for test
    param = new matmul_param;
    param->height1 = height1;
    param->height2 = height2;
    param->width1 = width1;
    param->width2 = width2;
    param->outputsize = height1*width2;
    param->arraySize1 = width1*height1;
    param->arraySize2 = width2*height2;
    A1 = new float[param->arraySize1];
    A2 = new unsigned char[param->arraySize2];
    A3 = new float[param->outputsize];
    generateRandomFloatArray(A1, param->arraySize1);
    generateRandomCharArray(A2, param->arraySize2);
    // printArray(A1, param->arraySize1);
    // printArray(A2, param->arraySize2);
    

    // CPU
     auto start = high_resolution_clock::now();
     test_matmul(A1, param->height1,  param->width1, A2, param->height2, param->width2, A3);
     auto stop = high_resolution_clock::now();
     auto duration = duration_cast<microseconds>(stop - start);
     std::cout << "CPU: " << duration.count() << "ms" << std::endl;
     printf("CPU Results: \n");
     for (uint32_t i = 0; i < param->outputsize; i++){
         printf("A3[%d]: %f\n", i, A3[i]);
     }
     free(A3);
     A3 = new float[param->outputsize];
    
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

void test_matmulInt4(){
    // not considering offset atm
    Int4_buffer = new MetalMatmulBuffers;
    Int4_params = new MetalMatMulParams;
    Int4_params->group_size = group_size;
    Int4_params->height1 = height1; // m
    Int4_params->width1 = width1; // k
    Int4_params->width3 = width2; // n
    
    scales = new float[Int4_params->height1*Int4_params->width1/Int4_params->group_size];
    generateRandomFloatArray(scales, height1*width1/group_size);
    Int4_buffer->A = A1;
    Int4_buffer->B = A2;
    Int4_buffer->C = A3;
    Int4_buffer->scales = scales;
    metal_init();
    metal_compute();
    printf("GPU Results: \n");
    for (uint32_t i = 0; i < Int4_params->height1*Int4_params->width1/Int4_params->group_size; i++){
        printf("bM3[%d]: %f\n", i, ((float*)(bM3->contents()))[i]);
    }
}

void test_matmulInt4_simds(){
    fn_name = "matmulInt4_SIMD_Q4Interleave";
    Int4_Sbuffer = new MetalMatmulSBuffers;
    Int4_Sparams = new MetalMatMulParams;
    Int4_Sparams->group_size = group_size/4;
    Int4_Sparams->height1 = height1; // m
    Int4_Sparams->width1 = width1/4; // k
    Int4_Sparams->height2 = height2/2/4; // m
    Int4_Sparams->width2 = width2/2; // k
    Int4_Sparams->height3 = height1; // m
    Int4_Sparams->width3 = width2; // k
    
    scales = new float[Int4_Sparams->height2*Int4_Sparams->width2/Int4_Sparams->group_size];
    generateRandomFloatArray(scales, Int4_Sparams->height2*Int4_Sparams->width2/Int4_Sparams->group_size);
    Int4_Sbuffer->scales = scales;
    Int4_Sbuffer->C = A3;
    PackedFloat *tempF = new PackedFloat[Int4_Sparams->height1*Int4_Sparams->width1];
    PackedChar *tempC = new PackedChar[Int4_Sparams->width2*Int4_Sparams->width2];
    printf("GPU Results: 1 \n");
    for (size_t i = 0; i < height1*width1; i += 4) {
        for (int j = 0; j < 4; j++) {
            if (i + j < height1*width1) {  // Check to prevent out-of-bounds access
                tempF[i / 4].values[j] = A1[i + j];
            }
        }
    }
    Int4_Sbuffer->A = tempF;
    
    for (size_t i = 0; i < height2/2*width2/2; i += 4) {
        for (int j = 0; j < 4; j++) {
            if (i + j < height1*width1) {  // Check to prevent out-of-bounds access
                tempC[i / 4].values[j] = A2[i + j];
            }
        }
    }
    Int4_Sbuffer->B = tempC;
    
    metal_init();
    metal_compute();
    for (uint32_t i = 0; i < Int4_Sparams->height1*Int4_Sparams->width1/Int4_Sparams->group_size; i++){
        // for (int j = 0; j < 4; j++) {
        printf("bsM3[%d]: %f\n", i, ((float*)(bsM3->contents()))[i]);
        // }
    }
}

int main(){
    A1 = new float[height1*width1];
    A2 = new unsigned char[width1*width2];
    A3 = new float[height1*width2];
    scales = new float[height1*width1];
    generateRandomFloatArray(A1, height1*width1);
    generateRandomCharArray(A2, width1*width2);
    generateRandomScale(scales, height1*width1);
    // test_matmulInt4();
    test_matmulInt4_simds();
    return 0;
}



