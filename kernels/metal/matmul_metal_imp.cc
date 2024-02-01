#include "matmul_metal_imp.h"

#include <iostream>
// Some notes:
// 1. pipelinestate and encoder may be shared for improvement
// 2. since every time when a pointer is allocated, a metal buffer + its pointer will be
// associated together. In this case, the final result will be always stored in param.C
// 3. since metal is different from CUDA, it needs initialization and all Metal kernels
// should be placed in the same file, we place all metal kernels in the same kernel and
// all op helper functions here, which will be called later in ops. 

// static data
MTL::Device *MetalIMP::_mDevice;
MTL::ComputePipelineState *MetalIMP::_mMatmulFunctionPSO;
MTL::CommandQueue *MetalIMP::_mCommandQueue;

MTL::Buffer *MetalIMP::_mBufferA;
MTL::Buffer *MetalIMP::_mBufferB;
MTL::Buffer *MetalIMP::_mBufferScales;
MTL::Buffer *MetalIMP::_mBufferResult;
MTL::Buffer *MetalIMP::_mParams;

std::unordered_map<void *, MTL::Buffer *> MetalIMP::_mumap;

MetalMatMulParams *MetalIMP::_mParamsPtr;
bool MetalIMP::has_init = false;

void *MetalIMP::allocateSharedMem(size_t size) {
    if (!has_init) {
        MetalIMP::init();
    }

    MTL::Buffer *new_b = _mDevice->newBuffer(size, MTL::ResourceStorageModeShared);

    void *void_ptr = new_b->contents();

    // push the pair to the map
    _mumap.insert(std::make_pair(void_ptr, new_b));

    return void_ptr;
}


void MetalIMP::init() {
    _mDevice = MTL::CreateSystemDefaultDevice();
    has_init = true;
}

MTL::Buffer *MetalIMP::getBufferfromPtr(void *ptr) {
    if (_mumap.find(ptr) == _mumap.end()) {
        std::cerr << "Cannot find the corresponding MTL::Buffer." << std::endl;
        return NULL;
    } else
        return _mumap[ptr];
}

void MetalIMP::setupLibrary(const char *kernel_name){
    NS::Error *error = nullptr;

    // Load the shader files with a .metal file extension in the project
    MTL::Library *defaultLibrary = _mDevice->newDefaultLibrary();

    if (defaultLibrary == nullptr) {
        std::cout << "Failed to find the default library." << std::endl;
        return;
    }

    auto str = NS::String::string(kernel_name, NS::ASCIIStringEncoding);
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
}

void MetalIMP::SendEncode(MTL::Size gridSize, MTL::Size threadgroupSize, MTL::CommandBuffer *commandBuffer, MTL::ComputeCommandEncoder *computeEncoder){
    // Encode the compute command.
    computeEncoder->dispatchThreads(gridSize, threadgroupSize);

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

void MetalIMP::run_mat_mul_accelerator_int4_fast_no_offset(MetalMatMulParams param, MetalMatmulBuffers *bufferParams) {
    setupLibrary("matmulUInt4_SIMD_Q4Interleave_unroll32");

    _mParams = _mDevice->newBuffer(sizeof(MetalMatMulParams), MTL::ResourceStorageModeShared);
    _mParamsPtr = (MetalMatMulParams *)_mParams->contents();


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
    computeEncoder->setBuffer(_mBufferScales, 0, 3);
    computeEncoder->setBuffer(_mParams, 0, 4);

    MTL::Size gridSize = MTL::Size::Make(_mParamsPtr->n, _mParamsPtr->m, 1);

    // Calculate a threadgroup size.
    MTL::Size threadgroupSize = MTL::Size::Make(16, 1, 1);

    SendEncode(gridSize, threadgroupSize, commandBuffer, computeEncoder);
    _mMatmulFunctionPSO->release();
}

void MetalIMP::run_naive_mat_mul(MetalMatMulParams param, MetalMatmulBuffers *bufferParams) {
    setupLibrary("matmul");
    _mParams = _mDevice->newBuffer(sizeof(MetalMatMulParams), MTL::ResourceStorageModeShared);
    _mParamsPtr = (MetalMatMulParams *)_mParams->contents();


    *_mParamsPtr = param;
    unsigned int m, n, k;
    m = param.m;
    n = param.n;
    k = param.k;

    // assign the buffers to hold our data and the result.
    _mBufferA = getBufferfromPtr((void *)bufferParams->A);
    _mBufferB = getBufferfromPtr((void *)bufferParams->B);
    _mBufferResult = getBufferfromPtr((void *)bufferParams->C);

    if (!_mBufferA || !_mBufferB || !_mBufferResult) {
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

    MTL::Size threadgroupSize = MTL::Size::Make(8, 8, 1);
    MTL::Size gridSize = MTL::Size::Make((n + threadgroupSize.width - 1) / threadgroupSize.width,
                                (m + threadgroupSize.height - 1) / threadgroupSize.height,
                                              1);

    SendEncode(gridSize, threadgroupSize, commandBuffer, computeEncoder);
    _mMatmulFunctionPSO->release();
}

void MetalIMP::run_batch_add(MetalMatMulParams param, MetalMatmulBuffers *bufferParams){
    setupLibrary("kernel_batch_add");

    _mParams = _mDevice->newBuffer(sizeof(MetalMatMulParams), MTL::ResourceStorageModeShared);
    _mParamsPtr = (MetalMatMulParams *)_mParams->contents();


    *_mParamsPtr = param;
    unsigned int m, n, k;
    m = param.m;
    n = param.n;
    k = param.k;

    // assign the buffers to hold our data and the result.
    _mBufferA = getBufferfromPtr((void *)bufferParams->A);
    _mBufferB = getBufferfromPtr((void *)bufferParams->B);
    _mBufferResult = getBufferfromPtr((void *)bufferParams->C);

    if (!_mBufferA || !_mBufferB || !_mBufferResult) {
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

    MTL::Size threadgroupSize = MTL::Size::Make(8, 8, 1);
    MTL::Size gridSize = MTL::Size::Make((n + threadgroupSize.width - 1) / threadgroupSize.width,
                                (m + threadgroupSize.height - 1) / threadgroupSize.height,
                                              1);

    // Encode the compute command.
    computeEncoder->dispatchThreads(gridSize, threadgroupSize);

    // End the compute pass.
    computeEncoder->endEncoding();

    // Execute the command.
    commandBuffer->commit();

    // Normally, you want to do other work in your app while the GPU is running,
    // but in this example, the code simply blocks until the calculation is complete.
    commandBuffer->waitUntilCompleted();

    computeEncoder->release();
    commandBuffer->release();
    _mMatmulFunctionPSO->release();

}

void MetalIMP::run_relu(MetalMatMulParams param, MetalMatmulBuffers *bufferParams){
    setupLibrary("kernel_relu");

    _mParams = _mDevice->newBuffer(sizeof(MetalMatMulParams), MTL::ResourceStorageModeShared);
    _mParamsPtr = (MetalMatMulParams *)_mParams->contents();


    *_mParamsPtr = param;
    unsigned int m, n, k;
    m = param.m; // row1
    // n = param.n; // col2/3
    k = param.k; // col1

    // assign the buffers to hold our data and the result.
    _mBufferA = getBufferfromPtr((void *)bufferParams->A);
    _mBufferResult = getBufferfromPtr((void *)bufferParams->C);

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
    computeEncoder->setBuffer(_mBufferResult, 0, 1);

    MTL::Size threadgroupSize = MTL::Size::Make(8, 8, 1);
    MTL::Size gridSize = MTL::Size::Make((k + threadgroupSize.width - 1) / threadgroupSize.width,
                                (m + threadgroupSize.height - 1) / threadgroupSize.height,
                                              1);

    SendEncode(gridSize, threadgroupSize, commandBuffer, computeEncoder);
    _mMatmulFunctionPSO->release();
}

void MetalIMP::run_silu(MetalMatMulParams param, MetalMatmulBuffers *bufferParams){
    setupLibrary("kernel_silu");

    _mParams = _mDevice->newBuffer(sizeof(MetalMatMulParams), MTL::ResourceStorageModeShared);
    _mParamsPtr = (MetalMatMulParams *)_mParams->contents();


    *_mParamsPtr = param;
    unsigned int m, n, k;
    m = param.m; // row1
    // n = param.n; // col2/3
    k = param.k; // col1

    // assign the buffers to hold our data and the result.
    _mBufferA = getBufferfromPtr((void *)bufferParams->A);
    _mBufferResult = getBufferfromPtr((void *)bufferParams->C);

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
    computeEncoder->setBuffer(_mBufferResult, 0, 1);

    MTL::Size threadgroupSize = MTL::Size::Make(8, 8, 1);
    MTL::Size gridSize = MTL::Size::Make((k + threadgroupSize.width - 1) / threadgroupSize.width,
                                (m + threadgroupSize.height - 1) / threadgroupSize.height,
                                              1);

    SendEncode(gridSize, threadgroupSize, commandBuffer, computeEncoder);
    _mMatmulFunctionPSO->release();
}

void MetalIMP::run_gelu(MetalMatMulParams param, MetalMatmulBuffers *bufferParams){
    setupLibrary("kernel_gelu");

    _mParams = _mDevice->newBuffer(sizeof(MetalMatMulParams), MTL::ResourceStorageModeShared);
    _mParamsPtr = (MetalMatMulParams *)_mParams->contents();


    *_mParamsPtr = param;
    unsigned int m, n, k;
    m = param.m; // row1
    // n = param.n; // col2/3
    k = param.k; // col1

    // assign the buffers to hold our data and the result.
    _mBufferA = getBufferfromPtr((void *)bufferParams->A);
    _mBufferResult = getBufferfromPtr((void *)bufferParams->C);

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
    computeEncoder->setBuffer(_mBufferResult, 0, 1);

    MTL::Size threadgroupSize = MTL::Size::Make(8, 8, 1);
    MTL::Size gridSize = MTL::Size::Make((k + threadgroupSize.width - 1) / threadgroupSize.width,
                                (m + threadgroupSize.height - 1) / threadgroupSize.height,
                                              1);

    SendEncode(gridSize, threadgroupSize, commandBuffer, computeEncoder);
    _mMatmulFunctionPSO->release();
}


void MetalIMP::run_gelu_quick(MetalMatMulParams param, MetalMatmulBuffers *bufferParams){
    setupLibrary("kernel_gelu_quick");

    _mParams = _mDevice->newBuffer(sizeof(MetalMatMulParams), MTL::ResourceStorageModeShared);
    _mParamsPtr = (MetalMatMulParams *)_mParams->contents();


    *_mParamsPtr = param;
    unsigned int m, n, k;
    m = param.m; // row1
    // n = param.n; // col2/3
    k = param.k; // col1

    // assign the buffers to hold our data and the result.
    _mBufferA = getBufferfromPtr((void *)bufferParams->A);
    _mBufferResult = getBufferfromPtr((void *)bufferParams->C);

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
    computeEncoder->setBuffer(_mBufferResult, 0, 1);

    MTL::Size threadgroupSize = MTL::Size::Make(8, 8, 1);
    MTL::Size gridSize = MTL::Size::Make((k + threadgroupSize.width - 1) / threadgroupSize.width,
                                (m + threadgroupSize.height - 1) / threadgroupSize.height,
                                              1);

    SendEncode(gridSize, threadgroupSize, commandBuffer, computeEncoder);
    _mMatmulFunctionPSO->release();
}