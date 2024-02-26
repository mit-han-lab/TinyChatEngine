#include <cmath>
#include <iomanip>
#include "operators.h"
#include "utils.h"
#include "matmul_metal_imp.h"

// TODO: modify metal for weights
void LlamaRMSNorm_metal::forward(const Matrix3D<half> &x, Matrix3D<half> &output, float eps) {
    int m = x.m_dim_x * x.m_dim_y;
    int n = x.m_dim_z;
    dim3 grid(m);
    dim3 block(min(n, 1024));

    /* For general cases, n is equal to hidden_units, e.g., 512/1024.
       Since we have warp shuffle inside the code, block.x % 32 should be 0.
    */
    if (n % 32 != 0) {
        block.x = 1024;
    }

    block.x = block.x / (4 / sizeof(half));  // if using half, only need half of block.x

    setupLibrary("kernel_rms_norm");

    _mParams = _mDevice->newBuffer(sizeof(MetalMatMulParams), MTL::ResourceStorageModeShared);
    _mParamsPtr = (MetalMatMulParams *)_mParams->contents();
    _mParamsPtr->m_dim_x = x.m_dim_x;
    _mParamsPtr->m_dim_y = x.m_dim_y;
    _mParamsPtr->m_dim_z = x.m_dim_z;
    _mParamsPtr->eps = eps;
    _mParamsPtr->type_size = sizeof(half);


    /* should pay attention to the rsqrt precision */
    half *input = x.m_data, *out = output.m_data;
    float *gamma = weight.m_data;

    _mBufferA = getBufferfromPtr((void *)input);
    _mBufferB = getBufferfromPtr((void *)gamma);
    _mBufferResult = getBufferfromPtr((void *)out);



    if (!_mBufferA || !_mBufferResult) {
        std::cerr << "Failed to locate some buffer!" << std::endl;
        exit(-1);
    }

    // Create a command buffer to hold commands.
    MTL::CommandBuffer *commandBuffer = _mCommandQueue->commandBuffer();
    assert(commandBuffer != nullptr);

    // Start a compute pass.
    MTL::ComputeCommandEncoder *computeEncoder = commandBuffer->computeCommandEncoder();
    assert(computeEncoder != nullptr);

    // Encode the pipeline state object and its parameters.
    computeEncoder->setComputePipelineState(_mMatmulFunctionPSO);
    computeEncoder->setBuffer(_mBufferA, 0, 0);
    computeEncoder->setBuffer(_mBufferB, 0, 1);
    computeEncoder->setBuffer(_mBufferResult, 0, 2);
    computeEncoder->setBuffer(_mParams, 0, 3);

    computeEncoder->setThreadgroupMemoryLength(param.type_size * N_SIMDWIDTH, 0);

    MTL::Size threadgroupSize = MTL::Size::Make(block.x, block.y, block.z);
    MTL::Size gridSize = MTL::Size::Make(grid.x, grid.y, grid.z);

    SendEncode(gridSize, threadgroupSize, commandBuffer, computeEncoder);
    _mMatmulFunctionPSO->release();
}