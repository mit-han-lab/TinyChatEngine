#include "operators.h"
#include "utils.h"
#include "matmul_metal_imp.h"

void load_Embedding_params_metal(Embedding_cuda& op, std::string prefix) {
    op.lookup.load((prefix + "/weight.bin").c_str());
}

// TODO: implement metal side
void Embedding_cuda::forward(Matrix3D<int> input_id, Matrix3D<half> output) {
    PROFILE_START(profile_name);
    assert(input_id.m_dim_x == 1);
    assert(input_id.m_dim_y == 1);
    assert(input_id.m_dim_z == output.m_dim_y);
    assert(output.m_dim_z == this->embed_dim);

    int threadsPerBlock = 1024;
    int blocksPerGrid = (input_id.m_dim_z + threadsPerBlock - 1) / threadsPerBlock;

    setupLibrary("EmbeddingKernel");

    _mBufferA = getBufferfromPtr((void *)input_id);
    _mBufferB = getBufferfromPtr((void *)this->lookup.m_data);
    _mBufferResult = getBufferfromPtr((void *)output);
    _mBufferEmbed_dim = getBufferfromPtr((void *)this->embed_dim);

    if (!_mBufferA || !_mBufferB || !_mBufferResult || !_mBufferScales) {
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
    computeEncoder->setBuffer(_mBufferResult, 0, 1);
    computeEncoder->setBuffer(_mBufferB, 0, 2);
    computeEncoder->setBuffer(_mBufferEmbed_dim, 0, 3);

    MTL::Size gridSize = MTL::Size::Make(blocksPerGrid, 1, 1);

    // Calculate a threadgroup size
    MTL::Size threadgroupSize = MTL::Size::Make(threadsPerBlock, 1, 1);

    SendEncode(gridSize, threadgroupSize, commandBuffer, computeEncoder);
    _mMatmulFunctionPSO->release();


    PROFILE_END(profile_name);
}