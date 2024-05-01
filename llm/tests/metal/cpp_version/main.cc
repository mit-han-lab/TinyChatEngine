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

// .h
MTL::Buffer *bM1, *bM2, *bM3, *bParam, *bScales, *bOffset;
MTL::Device* mDevice;
MTL::ComputePipelineState* mfnPipelineState;
MTL::CommandQueue* mCommandQueue;
NS::Error *error = nullptr;
typedef struct {
    float *A, *C, *scales, *offset;
    unsigned char *B;
} MetalMatmulBuffers;


using namespace std;
using namespace chrono;

// .cc
const char * fn_name = "matmul";


// main
unsigned int height1 = 320*320;
unsigned int width1 = 320;
unsigned int height2 = 320;
unsigned int width2 = 320*320;
float *A1, *Anorm, *A3;
unsigned char *A2;
matmul_param *param;
// for MatmulInt4 use
unsigned int group_size = 32;
float* scales, *offset;
MetalMatmulBuffers *Int4_buffer;
MetalMatMulParams *Int4_params;

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
    std::uniform_real_distribution<float> distribution(0, 1);

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
void generateOnesCharArray(unsigned char* array, uint arraySize) {
    for (int i = 0; i < arraySize; ++i) {
        array[i] = 1;
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

void metal_encodecommand_matmulInt4(MTL::ComputeCommandEncoder *computeEncoder){
    //Create Metal buffers for input and output, if inside the TinyChat, param should be created in advance

    bScales = metal_newBuf(sizeof(float), (Int4_params->width2*Int4_params->width1+Int4_params->group_size-1)/Int4_params->group_size);
    bM1 = metal_newBuf(sizeof(float), Int4_params->arraySize1);
    bM2 = metal_newBuf(sizeof(unsigned char), Int4_params->arraySize2);
    bParam = metal_newBuf(sizeof(MetalMatMulParams), 1);
    bM3 = metal_newBuf(sizeof(float), Int4_params->outputsize);
    
    computeEncoder->setComputePipelineState(mfnPipelineState);
    computeEncoder->setBuffer(bM1, 0, 0);
    computeEncoder->setBuffer(bM2, 0, 1);
    computeEncoder->setBuffer(bM3, 0, 2);
    computeEncoder->setBuffer(bScales, 0, 3);
    computeEncoder->setBuffer(bParam, 0, 4);
    
    memcpy(bM1->contents(), Int4_buffer->A, Int4_params->arraySize1*sizeof(float));
    memcpy(bM2->contents(), Int4_buffer->B, Int4_params->arraySize2*sizeof(unsigned char));
    memcpy(bM3->contents(), Int4_buffer->C, Int4_params->outputsize*sizeof(float));
    memcpy(bParam->contents(), Int4_params, sizeof(MetalMatMulParams));
    memcpy(bScales->contents(), Int4_buffer->scales, (Int4_params->width2*Int4_params->width1+Int4_params->group_size-1)/Int4_params->group_size*sizeof(float));
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
    memcpy(bM2->contents(), Anorm, param->arraySize2*sizeof(float));
    memcpy(bM3->contents(), A3, param->outputsize*sizeof(float));
    memcpy(bParam->contents(), param, sizeof(matmul_param));
}

void metal_compute(){
    // Initialization of GPU vals
    MTL::CommandBuffer *commandBuffer = mCommandQueue->commandBuffer();
    assert(commandBuffer != nullptr);
    MTL::ComputeCommandEncoder *computeEncoder = commandBuffer->computeCommandEncoder();
    assert(computeEncoder != nullptr);
    
    if (strcmp(fn_name, "matmul") == 0) {
        metal_encodecommand_matmul(computeEncoder);
    } else {
        metal_encodecommand_matmulInt4(computeEncoder);
    }
    
    // Threads -> ThreadGroup -> Grid
    MTL::Size mThreadGroupSize;
    MTL::Size mGridSize;
    if (strcmp(fn_name, "matmul") == 0) {
        auto threadsPerThreadgroup = mDevice->maxThreadsPerThreadgroup();
        mThreadGroupSize = MTL::Size::Make(32, 32, 1);
        mGridSize = MTL::Size::Make((param->width2 + mThreadGroupSize.width - 1) / mThreadGroupSize.width,
                                              (param->height1 + mThreadGroupSize.height - 1) / mThreadGroupSize.height,
                                              1);
    } else {
        auto threadsPerThreadgroup = mDevice->maxThreadsPerThreadgroup();
        // for test Normal Matmul (16, 16, 1); 
        // for test Int4, (16, 1, 1)
        mThreadGroupSize = MTL::Size::Make(1, 9, 1); // for test, (16, 1, 1)
        mGridSize = MTL::Size::Make((Int4_params->width3+ mThreadGroupSize.width - 1)/mThreadGroupSize.width, (Int4_params->height1+ mThreadGroupSize.height - 1)/mThreadGroupSize.height, 1);
    }
    
    // Dispatch and Run Computation
    
    computeEncoder->dispatchThreadgroups(mGridSize, mThreadGroupSize);
    computeEncoder->endEncoding();
    commandBuffer->commit();
    auto start = high_resolution_clock::now();
    commandBuffer->waitUntilCompleted();
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    std::cout << duration.count() << std::endl;
    computeEncoder->release();
    commandBuffer->release();
}

void metal_rms_compute(){
    // Initialization of GPU vals
    MTL::CommandBuffer *commandBuffer = mCommandQueue->commandBuffer();
    assert(commandBuffer != nullptr);
    MTL::ComputeCommandEncoder *computeEncoder = commandBuffer->computeCommandEncoder();
    assert(computeEncoder != nullptr);
    
    bM1 = metal_newBuf(sizeof(float), Int4_params->arraySize1);
    bM3 = metal_newBuf(sizeof(float), Int4_params->outputsize);
    bParam = metal_newBuf(sizeof(MetalMatMulParams), 1);
    
    computeEncoder->setComputePipelineState(mfnPipelineState);
    computeEncoder->setBuffer(bM1, 0, 0);
    computeEncoder->setBuffer(bM3, 0, 1);
    computeEncoder->setBuffer(bParam, 0, 2);

    computeEncoder->setThreadgroupMemoryLength(Int4_params->type_size * group_size, 0);
    
    memcpy(bM1->contents(), A1, Int4_params->arraySize1*sizeof(float));
    memcpy(bParam->contents(), Int4_params, sizeof(MetalMatMulParams));
    
    // Threads -> ThreadGroup -> Grid
   
    MTL::Size threadgroupSize = MTL::Size::Make(MIN(16, Int4_params->m_dim_z), MIN(16, Int4_params->m_dim_y), 1);
    MTL::Size gridSize = MTL::Size::Make((Int4_params->m_dim_z + threadgroupSize.width - 1) / threadgroupSize.width,
                                (Int4_params->m_dim_y + threadgroupSize.height - 1) / threadgroupSize.height,
                                              1);
    
    // Dispatch and Run Computation
    auto start = high_resolution_clock::now();
    computeEncoder->dispatchThreadgroups(gridSize, threadgroupSize);
    computeEncoder->endEncoding();
    commandBuffer->commit();
    
    commandBuffer->waitUntilCompleted();
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    std::cout << duration.count() << std::endl;
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
    Anorm = new float[param->arraySize2];
    A3 = new float[param->outputsize];
    generateRandomFloatArray(A1, param->arraySize1);
    generateRandomFloatArray(Anorm, param->arraySize2);
    // printArray(A1, param->arraySize1);
    // printArray(A2, param->arraySize2);
    

    // // CPU
    //  auto start = high_resolution_clock::now();
    //  test_matmul(A1, param->height1,  param->width1, A2, param->height2, param->width2, A3);
    //  auto stop = high_resolution_clock::now();
    //  auto duration = duration_cast<microseconds>(stop - start);
    //  std::cout << "CPU: " << duration.count() << "ms" << std::endl;
    //  printf("CPU Results: \n");
    //  for (uint32_t i = 0; i < param->outputsize; i++){
    //      printf("A3[%d]: %f\n", i, A3[i]);
    //  }
    //  free(A3);
    //  A3 = new float[param->outputsize];
    
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
    fn_name = "matmulInt4"; 
    fn_name = "matmulUInt4_SIMD_Q4Interleave_unroll32";
    Int4_buffer = new MetalMatmulBuffers;
    Int4_params = new MetalMatMulParams;
    Int4_params->group_size = group_size;
    Int4_params->height1 = height1; // m
    Int4_params->width1 = width1; // k
    Int4_params->height2 = height2; // n
    Int4_params->width2 = width2;
    Int4_params->height3 = height1;
    Int4_params->width3 = width2;

    Int4_params-> arraySize1 = Int4_params->height1*Int4_params->width1;
    Int4_params-> arraySize2 = Int4_params->height2*Int4_params->width2;
    Int4_params-> outputsize = Int4_params->height3*Int4_params->width3;
    A1 = new float[Int4_params-> arraySize1];
    A2 = new unsigned char[Int4_params-> arraySize2];
    A3 = new float[Int4_params-> outputsize];
    scales = new float[(Int4_params->width2*Int4_params->width1+Int4_params->group_size-1)/Int4_params->group_size];
    generateRandomFloatArray(A1, Int4_params-> arraySize1);
    generateRandomCharArray(A2, Int4_params-> arraySize2);
    generateRandomFloatArray(scales, (Int4_params->width2*Int4_params->width1+Int4_params->group_size-1)/Int4_params->group_size);
    Int4_buffer->A = A1;
    Int4_buffer->B = A2;
    Int4_buffer->C = A3;
    Int4_buffer->scales = scales;
    metal_init();
    metal_compute();
    // for (uint32_t i = 0; i < Int4_params-> outputsize; i++){
    //     printf("bM3[%d]: %f\n", i, ((float*)(bM3->contents()))[i]);
    // }
}

void test_rms_nor(){
    fn_name = "kernel_rms_norm";
    Int4_buffer = new MetalMatmulBuffers;
    Int4_params = new MetalMatMulParams;
    Int4_params->group_size = group_size;
    Int4_params->height1 = 960; // m
    Int4_params->width1 = 4096; // k
    Int4_params->height2 = 960; // n
    Int4_params->width2 = 4096;
    Int4_params->height3 = 960;
    Int4_params->width3 = 4096;
    Int4_params-> arraySize1 = Int4_params->height1*Int4_params->width1;
    Int4_params-> arraySize2 = Int4_params->height2*Int4_params->width2;
    Int4_params-> outputsize = Int4_params->height3*Int4_params->width3;
    A1 = new float[Int4_params-> arraySize1];
    A3 = new float[Int4_params-> outputsize];
    generateRandomFloatArray(A1, Int4_params-> arraySize1);
    Int4_params->m_dim_x = 1;
    Int4_params->m_dim_y = Int4_params->height1;
    Int4_params->m_dim_z = Int4_params->width1;
    Int4_params->eps = 1e-06;
    Int4_params->type_size = sizeof(float);
    metal_init();
    metal_rms_compute();
    
}

void test_matmul_llama(){
    // m: 1, n = 32000, k = 4096 (lm_head)
    // m: 1, n = 4096, k = 4096 (Q, K, V, out projections)
    // m: 1, n = 4096, k = 11008 (down_proj)
    // m: 1, n = 11008, k = 4096 (up_proj and gate_proj)

    // in ggml doc: https://github.com/ggerganov/whisper.cpp/blob/master/ggml.h
    // ne[GGML_MAX_DIMS] => number of elements
        // ne10 => number of elements of src1 along dim_0
    // nb[GGML_MAX_DIMS] => stride in bytes:
        // nb[0] = ggml_type_size(type)
        // nb[1] = nb[0]   * (ne[0] / ggml_blck_size(type)) + padding
        // nb[i] = nb[i-1] * ne[i-1]

    fn_name = "kernel_mul_mm_impl";
    int bs = 1;
    int m = 1;
    int n = 32000;
    int k = 4096;
    int block_size = 32;
    int hidden_size = bs * m * k;
    int weight_size = bs * n * k;
    int output_size = bs * m * n;
    unsigned char* src0 = new unsigned char[hidden_size];
    unsigned char* src1 = new unsigned char[weight_size];
    float* dst = new float[output_size];
    generateRandomCharArray(src0, hidden_size);
    generateRandomCharArray(src1, weight_size);
    // generateOnesCharArray(src0, hidden_size);
    // generateOnesCharArray(src1, weight_size);
    // generateRandomFloatArray(dst, arraySize);
    metal_init();
    // Initialization of GPU vals
    MTL::CommandBuffer *commandBuffer = mCommandQueue->commandBuffer();
    assert(commandBuffer != nullptr);
    MTL::ComputeCommandEncoder *computeEncoder = commandBuffer->computeCommandEncoder();
    assert(computeEncoder != nullptr);

    bM1 = metal_newBuf(sizeof(unsigned char), hidden_size);
    bM2 = metal_newBuf(sizeof(unsigned char), weight_size);
    bM3 = metal_newBuf(sizeof(float), output_size);
    MTL::Buffer *bne00 = metal_newBuf(sizeof(int64_t), 1);
    MTL::Buffer *bne02 = metal_newBuf(sizeof(int64_t), 1);
    MTL::Buffer *bnb01 = metal_newBuf(sizeof(uint64_t), 1);
    MTL::Buffer *bnb02 = metal_newBuf(sizeof(uint64_t), 1);
    MTL::Buffer *bne12 = metal_newBuf(sizeof(int64_t), 1);
    MTL::Buffer *bnb10 = metal_newBuf(sizeof(uint64_t), 1);
    MTL::Buffer *bnb11 = metal_newBuf(sizeof(uint64_t), 1);
    MTL::Buffer *bnb12 = metal_newBuf(sizeof(uint64_t), 1);
    MTL::Buffer *bne0 = metal_newBuf(sizeof(int64_t), 1);
    MTL::Buffer *bne1 = metal_newBuf(sizeof(int64_t), 1);
    MTL::Buffer *br2 = metal_newBuf(sizeof(uint), 1);
    MTL::Buffer *br3 = metal_newBuf(sizeof(uint), 1);

    computeEncoder->setComputePipelineState(mfnPipelineState);
    computeEncoder->setBuffer(bM1, 0, 0);
    computeEncoder->setBuffer(bM2, 0, 1);
    computeEncoder->setBuffer(bM3, 0, 2);
    computeEncoder->setBuffer(bne00, 0, 3);
    computeEncoder->setBuffer(bne02, 0, 4);
    computeEncoder->setBuffer(bnb01, 0, 5);
    computeEncoder->setBuffer(bnb02, 0, 6);
    computeEncoder->setBuffer(bne12, 0, 7);
    computeEncoder->setBuffer(bnb10, 0, 8);
    computeEncoder->setBuffer(bnb11, 0, 9);
    computeEncoder->setBuffer(bnb12, 0, 10);
    computeEncoder->setBuffer(bne0, 0, 11);
    computeEncoder->setBuffer(bne1, 0, 12);
    computeEncoder->setBuffer(br2, 0, 13);
    computeEncoder->setBuffer(br3, 0, 14);
    computeEncoder->setThreadgroupMemoryLength(8192, 0); // from https://github.com/ggerganov/llama.cpp/blob/d5ab29757ebc59a30f03e408294ec20628a6374e/ggml-metal.m#L1315

    
    int64_t ne00 = k;
    int64_t ne01 = n;
    int64_t ne02 = bs;
    int64_t ne03 = 1;
    uint64_t nb00 = sizeof(unsigned char);
    uint64_t nb01 = nb00 * ne00 / block_size;  // nb[0] * (ne[0] / ggml_blck_size(type)) + padding; BUG: ggml_blck_size
    uint64_t nb02 = nb01 * ne01;
    int64_t ne10 = k;
    int64_t ne11 = m;
    int64_t ne12 = bs;
    int64_t ne13 = 1;
    uint64_t nb10 = sizeof(unsigned char);
    uint64_t nb11 = nb10 * ne10;
    uint64_t nb12 = nb11 * ne11;
    int64_t ne0 = n;
    int64_t ne1 = m;
    uint r2 = ne12 / ne02;
    uint r3 = ne13 / ne03;
    memcpy(bM1->contents(), src0, hidden_size * sizeof(unsigned char));
    memcpy(bM2->contents(), src1, weight_size * sizeof(unsigned char));
    memcpy(bM3->contents(), dst, output_size * sizeof(float));
    memcpy(bne00->contents(), &ne00, sizeof(ne00));
    memcpy(bne02->contents(), &ne02, sizeof(ne02));
    memcpy(bnb01->contents(), &nb01, sizeof(nb01));
    memcpy(bnb02->contents(), &nb02, sizeof(nb02));
    memcpy(bne12->contents(), &ne12, sizeof(ne12));
    memcpy(bnb10->contents(), &nb10, sizeof(nb10));
    memcpy(bnb11->contents(), &nb11, sizeof(nb11));
    memcpy(bnb12->contents(), &nb12, sizeof(nb12));
    memcpy(bne0->contents(), &ne0, sizeof(ne0));
    memcpy(bne1->contents(), &ne1, sizeof(ne1));
    memcpy(br2->contents(), &r2, sizeof(r2));
    memcpy(br3->contents(), &r3, sizeof(r3));

    std::cout << "src0: ";
    for (int i = 0; i < 10; ++i) {
        std::cout << src0[i] << " ";
    }
    std::cout << std::endl;
    std::cout << "bM1: ";
    for (int i = 0; i < 10; ++i) {
        std::cout << ((unsigned char*)(bM1->contents()))[i] << " ";
    }
    std::cout << std::endl;
    
    // Assuming you have already configured the threadgroup size and number of threadgroups based on your kernel and data
    MTL::Size threadgroupSize = MTL::Size::Make(128, 1, 1);
    MTL::Size numThreadgroups = MTL::Size::Make((ne11 + 31)/32, (ne01 + 63)/64, ne12*ne13); // from https://github.com/ggerganov/llama.cpp/blob/d5ab29757ebc59a30f03e408294ec20628a6374e/ggml-metal.m#L1405
    
    // Dispatch the kernel
    computeEncoder->dispatchThreadgroups(numThreadgroups, threadgroupSize);
    
    // Finish encoding and commit the command buffer
    computeEncoder->endEncoding();
    commandBuffer->commit();
    auto start = high_resolution_clock::now();
    commandBuffer->waitUntilCompleted();
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    std::cout << "Metal GPU Duration: " << duration.count() << " ms" << std::endl;

    memcpy(dst, bM3->contents(), output_size * sizeof(float));

    // print dst
    std::cout << "dst: ";
    for (int i = 0; i < 10; ++i) {
        std::cout << dst[i] << " ";
    }
    // for (int i = 0; i < output_size; ++i) {
    //     if (dst[i] != 0) {
    //         std::cout << dst[i] << " ";
    //     }
    // }
    std::cout << std::endl;

    // print bM3
    std::cout << "bM3: ";
    for (int i = 0; i < 10; ++i) {
        std::cout << ((float*)(bM3->contents()))[i] << " ";
    }
    // for (int i = 0; i < output_size; ++i) {
    //     if (((float*)(bM3->contents()))[i] != 0) {
    //         std::cout << ((float*)(bM3->contents()))[i] << " ";
    //     }
    // }
    std::cout << std::endl;
}

int main(){
    test_matmul_llama();
    // test_matmulInt4();
    return 0;
}



