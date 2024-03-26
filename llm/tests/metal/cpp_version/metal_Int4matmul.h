#pragma once

#include "Foundation/Foundation.hpp"
#include "Metal/Metal.hpp"
#include "param.h"

class MetalMatmulInt4IMP {
   public:
    MTL::Buffer *bM1, *bM2, *bM3, *bParam, *bScales, *bOffset;
    MTL::Device* mDevice;
    MTL::ComputePipelineState* mfnPipelineState;
    MTL::CommandQueue* mCommandQueue;
    NS::Error *error = nullptr;
    typedef struct {
        float *A, *C, *scales, *offset;
        unsigned char *B;
    } MetalMatmulBuffers;

    void metal_init();
    void metal_encodecommand(MTL::ComputeCommandEncoder *computeEncoder);
    void metal_compute();
    MTL::Buffer *metal_newBuf(unsigned long type_size, unsigned long size);
};
