#pragma once

#include "Foundation/Foundation.hpp"
#include "Metal/Metal.hpp"
#include "opParams.h"


class MetalMatmul
{
public:
    MTL::Device *_mDevice;

    // The compute pipeline generated from the compute kernel in the .metal shader file.
    MTL::ComputePipelineState *_mMatmulFunctionPSO;

    // The command queue used to pass commands to the device.
    MTL::CommandQueue *_mCommandQueue;

    // Buffers to hold data.
    MTL::Buffer *_mBufferA;
    MTL::Buffer *_mBufferB;
    MTL::Buffer *_mBufferResult;
    MTL::Buffer *_mParams;

    // Matmul params
    MatMulParams *_mParamsPtr;

    MetalMatmul(MTL::Device *device, MatMulParams param);
    ~MetalMatmul();

    void prepareData();
    void sendComputeCommand();
    void verifyResults();

private:
    void encodeCommand(MTL::ComputeCommandEncoder *computeEncoder);
    void generateRandomFloatData(MTL::Buffer *buffer, int length);
};
