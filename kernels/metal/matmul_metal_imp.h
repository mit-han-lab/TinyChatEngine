#pragma once

#include <unordered_map>

#include "Foundation/Foundation.hpp"
#include "Metal/Metal.hpp"
#include "include/opParams.h"

typedef struct {
    float *A, *C, *scales, *offset;
    unsigned char *B;
} MetalMatmulBuffers;

class MetalIMP {
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
    static void setupLibrary(const char *kernel_name);
    static void *allocateSharedMem(size_t size);
    static MetalMatMulParams *_mParamsPtr;
    static MTL::Buffer *getBufferfromPtr(void *ptr);

    static void run_mat_mul_accelerator_int4_fast_no_offset(MetalMatMulParams param, MetalMatmulBuffers *bufferParams);
    static void run_naive_mat_mul(MetalMatMulParams param, MetalMatmulBuffers *bufferParams);
    static void run_batch_add(MetalMatMulParams param, MetalMatmulBuffers *bufferParams);
    static void run_relu(MetalMatMulParams param, MetalMatmulBuffers *bufferParams);
    static void run_silu(MetalMatMulParams param, MetalMatmulBuffers *bufferParams);
    static void run_gelu(MetalMatMulParams param, MetalMatmulBuffers *bufferParams);
    static void run_gelu_quick(MetalMatMulParams param, MetalMatmulBuffers *bufferParams);
    static void run_rms_norm(MetalMatMulParams param, MetalMatmulBuffers *bufferParams); // TODO: to be fixed
    static void run_soft_max(MetalMatMulParams param, MetalMatmulBuffers *bufferParams); // TODO: to be fixed
    static void run_soft_max_4(MetalMatMulParams param, MetalMatmulBuffers *bufferParams); // TODO: to be fixed
    static void run_rope(MetalMatMulParams param, MetalMatmulBuffers *bufferParams); // TODO: to be fixed


    // static void sendComputeCommand();
    // static void encodeCommand(MTL::ComputeCommandEncoder *computeEncoder);
};
