#pragma once

#include <unordered_map>

#include "Foundation/Foundation.hpp"
#include "Metal/Metal.hpp"
#include "include/opParams.h"

typedef struct {
    float *A, *C, *scales, *offset;
    unsigned char *B;
} MetalMatmulBuffers;

class MetalMatmulInt4IMP {
   public:
    static MTL::Device *_mDevice;

    // The compute pipeline generated from the compute kernel in the .metal shader file.
    static MTL::ComputePipelineState *_mMatmulFunctionPSO;

    // The command queue used to pass commands to the device.
    static MTL::CommandQueue *_mCommandQueue;

    // Buffers to hold data.
    static MTL::Buffer *_mBufferA;
    static MTL::Buffer *_mBufferB;
    static MTL::Buffer *_mBufferScales;
    static MTL::Buffer *_mBufferResult;
    static MTL::Buffer *_mParams;

    static std::unordered_map<void *, MTL::Buffer *> _mumap;

    static bool has_init;
    static void init();
    static void run(MetalMatMulParams param, MetalMatmulBuffers *bufferParams);
    static void *allocateSharedMem(size_t size);

    static MetalMatMulParams *_mParamsPtr;
    static void sendComputeCommand();
    static void encodeCommand(MTL::ComputeCommandEncoder *computeEncoder);
    static MTL::Buffer *getBufferfromPtr(void *ptr);
};
