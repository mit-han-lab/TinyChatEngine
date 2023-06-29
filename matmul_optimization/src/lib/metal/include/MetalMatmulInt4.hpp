#pragma once

#include "Foundation/Foundation.hpp"
#include "Metal/Metal.hpp"
#include "opParams.h"

class MetalMatmulInt4 {
   public:
    MTL::Device *_mDevice;

    // The compute pipeline generated from the compute kernel in the .metal shader file.
    MTL::ComputePipelineState *_mMatmulFunctionPSO;

    // The command queue used to pass commands to the device.
    MTL::CommandQueue *_mCommandQueue;

    // Buffers to hold data.
    MTL::Buffer *_mBufferA;
    MTL::Buffer *_mBufferB;
    MTL::Buffer *_mBufferScales;
    MTL::Buffer *_mBufferResult;
    MTL::Buffer *_mParams;

    // Matmul params
    MetalMatMulParams *_mParamsPtr;

    MetalMatmulInt4(MTL::Device *device, MetalMatMulParams param);
    ~MetalMatmulInt4();

    void prepareData();
    void sendComputeCommand();
    void verifyResults();

   private:
    void encodeCommand(MTL::ComputeCommandEncoder *computeEncoder);
    void generateRandomFloatData(MTL::Buffer *buffer, int length);
    void generateRandomIn4Data(MTL::Buffer *buffer, int length);
};
