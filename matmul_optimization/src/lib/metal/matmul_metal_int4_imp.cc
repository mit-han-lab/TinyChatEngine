#include "matmul_metal_int4_imp.h"

#include <iostream>

// static data
MTL::Device *MetalMatmulInt4IMP::_mDevice;
MTL::ComputePipelineState *MetalMatmulInt4IMP::_mMatmulFunctionPSO;
MTL::CommandQueue *MetalMatmulInt4IMP::_mCommandQueue;

MTL::Buffer *MetalMatmulInt4IMP::_mBufferA;
MTL::Buffer *MetalMatmulInt4IMP::_mBufferB;
MTL::Buffer *MetalMatmulInt4IMP::_mBufferScales;
MTL::Buffer *MetalMatmulInt4IMP::_mBufferResult;
MTL::Buffer *MetalMatmulInt4IMP::_mParams;

std::unordered_map<void *, MTL::Buffer *> MetalMatmulInt4IMP::_mumap;

MetalMatMulParams *MetalMatmulInt4IMP::_mParamsPtr;
bool MetalMatmulInt4IMP::has_init = false;

void *MetalMatmulInt4IMP::allocateSharedMem(size_t size) {
    if (!has_init) {
        MetalMatmulInt4IMP::init();
    }

    MTL::Buffer *new_b = _mDevice->newBuffer(size, MTL::ResourceStorageModeShared);

    void *void_ptr = new_b->contents();

    // push the pair to the map
    _mumap.insert(std::make_pair(void_ptr, new_b));

    return void_ptr;
}

void MetalMatmulInt4IMP::init() {
    _mDevice = MTL::CreateSystemDefaultDevice();
    ;

    NS::Error *error = nullptr;

    // Load the shader files with a .metal file extension in the project
    MTL::Library *defaultLibrary = _mDevice->newDefaultLibrary();

    if (defaultLibrary == nullptr) {
        std::cout << "Failed to find the default library." << std::endl;
        return;
    }

    auto str = NS::String::string("matmulUInt4_SIMD_Q4Interleave_unroll32", NS::ASCIIStringEncoding);
    MTL::Function *matmulFunction = defaultLibrary->newFunction(str);
    defaultLibrary->release();

    if (matmulFunction == nullptr) {
        std::cout << "Failed to find the function." << std::endl;
        return;
    }

    // Create a compute pipeline state object.
    _mMatmulFunctionPSO = _mDevice->newComputePipelineState(matmulFunction, &error);
    matmulFunction->release();

    if (_mMatmulFunctionPSO == nullptr) {
        //  If the Metal API validation is enabled, you can find out more information about what
        //  went wrong.  (Metal API validation is enabled by default when a debug build is run
        //  from Xcode)
        std::cout << "Failed to created pipeline state object, error " << error << "." << std::endl;
        return;
    }

    _mCommandQueue = _mDevice->newCommandQueue();
    if (_mCommandQueue == nullptr) {
        std::cout << "Failed to find the command queue." << std::endl;
        return;
    }

    _mParams = _mDevice->newBuffer(sizeof(MetalMatMulParams), MTL::ResourceStorageModeShared);
    _mParamsPtr = (MetalMatMulParams *)_mParams->contents();

    has_init = true;
}

MTL::Buffer *MetalMatmulInt4IMP::getBufferfromPtr(void *ptr) {
    if (_mumap.find(ptr) == _mumap.end()) {
        std::cerr << "Cannot find the corresponding MTL::Buffer." << std::endl;
        return NULL;
    } else
        return _mumap[ptr];
}

void MetalMatmulInt4IMP::run(MetalMatMulParams param, MetalMatmulBuffers *bufferParams) {
    *_mParamsPtr = param;
    unsigned int m, n, k;
    m = param.m;
    n = param.n;
    k = param.k;

    // assign the buffers to hold our data and the result.
    _mBufferA = getBufferfromPtr((void *)bufferParams->A);
    _mBufferB = getBufferfromPtr((void *)bufferParams->B);
    _mBufferResult = getBufferfromPtr((void *)bufferParams->C);
    _mBufferScales = getBufferfromPtr((void *)bufferParams->scales);

    if (!_mBufferA || !_mBufferB || !_mBufferResult || !_mBufferScales) {
        std::cerr << "Failed to locate some buffer!" << std::endl;
        exit(-1);
    }
    // TODO: offset?
    sendComputeCommand();
}

typedef std::chrono::microseconds time_unit;
void MetalMatmulInt4IMP::sendComputeCommand() {
    // Create a command buffer to hold commands.
    MTL::CommandBuffer *commandBuffer = _mCommandQueue->commandBuffer();
    assert(commandBuffer != nullptr);

    // Start a compute pass.
    MTL::ComputeCommandEncoder *computeEncoder = commandBuffer->computeCommandEncoder();
    assert(computeEncoder != nullptr);

    encodeCommand(computeEncoder);

    // End the compute pass.
    computeEncoder->endEncoding();

    // Execute the command.
    commandBuffer->commit();

    // Normally, you want to do other work in your app while the GPU is running,
    // but in this example, the code simply blocks until the calculation is complete.
    commandBuffer->waitUntilCompleted();

    computeEncoder->release();
    commandBuffer->release();
}

void MetalMatmulInt4IMP::encodeCommand(MTL::ComputeCommandEncoder *computeEncoder) {
    // Encode the pipeline state object and its parameters.
    computeEncoder->setComputePipelineState(_mMatmulFunctionPSO);
    computeEncoder->setBuffer(_mBufferA, 0, 0);
    computeEncoder->setBuffer(_mBufferB, 0, 1);
    computeEncoder->setBuffer(_mBufferResult, 0, 2);
    computeEncoder->setBuffer(_mBufferScales, 0, 3);
    computeEncoder->setBuffer(_mParams, 0, 4);

    MTL::Size gridSize = MTL::Size::Make(_mParamsPtr->n, _mParamsPtr->m, 1);

    // Calculate a threadgroup size.
    MTL::Size threadgroupSize = MTL::Size::Make(16, 1, 1);

    // Encode the compute command.
    computeEncoder->dispatchThreads(gridSize, threadgroupSize);
}
