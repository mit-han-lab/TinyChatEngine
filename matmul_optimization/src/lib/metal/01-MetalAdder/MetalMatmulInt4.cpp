#include "MetalMatmulInt4.hpp"
#include <iostream>

MetalMatmulInt4::MetalMatmulInt4(MTL::Device *device, MatMulParams param)
{
    _mDevice = device;

    NS::Error *error = nullptr;

    // Load the shader files with a .metal file extension in the project
    MTL::Library *defaultLibrary = _mDevice->newDefaultLibrary();

    if (defaultLibrary == nullptr)
    {
        std::cout << "Failed to find the default library." << std::endl;
        return;
    }

    auto str = NS::String::string("matmul", NS::ASCIIStringEncoding);
    MTL::Function *matmulFunction = defaultLibrary->newFunction(str);
    defaultLibrary->release();

    if (matmulFunction == nullptr)
    {
        std::cout << "Failed to find the function." << std::endl;
        return;
    }

    // Create a compute pipeline state object.
    _mMatmulFunctionPSO = _mDevice->newComputePipelineState(matmulFunction, &error);
    matmulFunction->release();

    if (_mMatmulFunctionPSO == nullptr)
    {
        //  If the Metal API validation is enabled, you can find out more information about what
        //  went wrong.  (Metal API validation is enabled by default when a debug build is run
        //  from Xcode)
        std::cout << "Failed to created pipeline state object, error " << error << "." << std::endl;
        return;
    }

    _mCommandQueue = _mDevice->newCommandQueue();
    if (_mCommandQueue == nullptr)
    {
        std::cout << "Failed to find the command queue." << std::endl;
        return;
    }

    // Allocate three buffers to hold our initial data and the result.
    _mBufferA = _mDevice->newBuffer(param.m * param.k * sizeof(float), MTL::ResourceStorageModeShared);
    _mBufferB = _mDevice->newBuffer(((param.n * param.k) / 2) * sizeof(uint8_t), MTL::ResourceStorageModeShared);
    _mBufferResult = _mDevice->newBuffer(param.m * param.n * sizeof(float), MTL::ResourceStorageModeShared);
    _mParams = _mDevice->newBuffer(sizeof(MatMulParams), MTL::ResourceStorageModeShared);

    _mParamsPtr = (MatMulParams*)_mParams->contents();
    *_mParamsPtr = param;

    printf("%d, %d, %d\n", _mParamsPtr->m, _mParamsPtr->n, _mParamsPtr->k);

    prepareData();
}

void MetalMatmulInt4::prepareData()
{
    generateRandomFloatData(_mBufferA, _mParamsPtr->m * _mParamsPtr-> k);
    generateRandomIn4Data(_mBufferB, _mParamsPtr->n * _mParamsPtr-> k);
}

typedef std::chrono::microseconds time_unit;
void MetalMatmulInt4::sendComputeCommand()
{
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
}

void MetalMatmulInt4::encodeCommand(MTL::ComputeCommandEncoder *computeEncoder)
{
    // Encode the pipeline state object and its parameters.
    computeEncoder->setComputePipelineState(_mMatmulFunctionPSO);
    computeEncoder->setBuffer(_mBufferA, 0, 0);
    computeEncoder->setBuffer(_mBufferB, 0, 1);
    computeEncoder->setBuffer(_mBufferResult, 0, 2);
    computeEncoder->setBuffer(_mParams, 0, 3);

    MTL::Size gridSize = MTL::Size::Make(_mParamsPtr->n, _mParamsPtr->m, 1);

    // Calculate a threadgroup size.
    MTL::Size threadgroupSize = MTL::Size::Make(64, 1, 1);

    // Encode the compute command.
    computeEncoder->dispatchThreads(gridSize, threadgroupSize);
}

void MetalMatmulInt4::generateRandomFloatData(MTL::Buffer *buffer, int length)
{
    float *dataPtr = (float *)buffer->contents();

    for (unsigned long index = 0; index < length; index++)
    {
        dataPtr[index] = (float)rand() / (float)(RAND_MAX);
    }
}

void MetalMatmulInt4::generateRandomIn4Data(MTL::Buffer *buffer, int length)
{
    uint8_t *dataPtr = (float *)buffer->contents();

    for (unsigned long index = 0; index < length / 2; index++)
    {
        uint8_t vl = (uint8_t)(((float)rand() / (float)(RAND_MAX)) * 15.0f);
        uint8_t vh = (uint8_t)(((float)rand() / (float)(RAND_MAX)) * 15.0f);
        dataPtr[index] = vl | (vh << 4);
    }
}

void MetalMatmulInt4::verifyResults()
{
    float *a = (float *)_mBufferA->contents();
    uint8_t *b = (float *)_mBufferB->contents();
    float *result = (float *)_mBufferResult->contents();

    assert(_mParamsPtr->n % 2 == 0);
    for (size_t i = 0; i < _mParamsPtr->m; i++){
        for (size_t j = 0; j < _mParamsPtr->n; j++){
            float sum = 0;
            for (size_t k = 0; k < _mParamsPtr->k; k+=2){
                size_t weight_idx = (j * _mParamsPtr->k + k) / 2;
                uint8_t weight_packed = b[weight_idx];
                uint8_t vl = b[weight_idx] & 0x0F;
                uint8_t vh = b[weight_idx] >> 4;

                sum += a[i * _mParamsPtr->k + k] * vl;
                sum += a[i * _mParamsPtr->k + k + 1] * vh;
            }
            float r = result[i * _mParamsPtr->n + j];
            if (std::abs(sum - r) > 1e-2){
                std::cout << "Expect " << sum << " at " << i << "," << j << ", but getting " << r << std::endl;
                throw("Result verification fails!");
            }
        }
    }
}

MetalMatmulInt4::~MetalMatmulInt4()
{
    _mBufferA->release();
    _mBufferB->release();
    _mBufferResult->release();

    _mMatmulFunctionPSO->release();
    _mCommandQueue->release();
}