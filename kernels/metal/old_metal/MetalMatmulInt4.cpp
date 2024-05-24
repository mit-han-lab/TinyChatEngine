#include "MetalMatmulInt4.hpp"

#include <iostream>

MetalMatmulInt4::MetalMatmulInt4(MTL::Device *device, MetalMatMulParams param) {
    _mDevice = device;

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

    // Allocate three buffers to hold our initial data and the result.
    _mBufferA = _mDevice->newBuffer(param.m * param.k * sizeof(float), MTL::ResourceStorageModeShared);
    _mBufferB = _mDevice->newBuffer(((param.n * param.k) / 2) * sizeof(uint8_t), MTL::ResourceStorageModeShared);
    _mBufferResult = _mDevice->newBuffer(param.m * param.n * sizeof(float), MTL::ResourceStorageModeShared);
    _mBufferScales =
        _mDevice->newBuffer(((param.n * param.k) / param.group_size) * sizeof(float), MTL::ResourceStorageModeShared);
    _mParams = _mDevice->newBuffer(sizeof(MetalMatMulParams), MTL::ResourceStorageModeShared);

    _mParamsPtr = (MetalMatMulParams *)_mParams->contents();
    *_mParamsPtr = param;

    printf("%d, %d, %d\n", _mParamsPtr->m, _mParamsPtr->n, _mParamsPtr->k);

    prepareData();
}

void MetalMatmulInt4::prepareData() {
    generateRandomFloatData(_mBufferA, _mParamsPtr->m * _mParamsPtr->k);
    generateRandomIn4Data(_mBufferB, _mParamsPtr->n * _mParamsPtr->k);
    generateRandomFloatData(_mBufferScales, (_mParamsPtr->n * _mParamsPtr->k) / _mParamsPtr->group_size);
}

typedef std::chrono::microseconds time_unit;
void MetalMatmulInt4::sendComputeCommand() {
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

void MetalMatmulInt4::encodeCommand(MTL::ComputeCommandEncoder *computeEncoder) {
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

void MetalMatmulInt4::generateRandomFloatData(MTL::Buffer *buffer, int length) {
    float *dataPtr = (float *)buffer->contents();

    for (unsigned long index = 0; index < length; index++) {
        dataPtr[index] = (float)rand() / (float)(RAND_MAX);
    }
}

void MetalMatmulInt4::generateRandomIn4Data(MTL::Buffer *buffer, int length) {
    int8_t *dataPtr = (int8_t *)buffer->contents();

    for (unsigned long index = 0; index < length / 2; index++) {
        int8_t vl = (int8_t)(((float)rand() / (float)(RAND_MAX)) * 15.0f) - 8;
        int8_t vh = (int8_t)(((float)rand() / (float)(RAND_MAX)) * 15.0f) - 8;
        dataPtr[index] = vl | (vh << 4);
    }
}

// ref: normal quantized format
// void MetalMatmulInt4::verifyResults()
// {
//     float *a = (float *)_mBufferA->contents();
//     uint8_t *b = (uint8_t *)_mBufferB->contents();
//     float *result = (float *)_mBufferResult->contents();
//     float *scales = (float *)_mBufferScales->contents();

//     assert(_mParamsPtr->n % 2 == 0);
//     for (size_t i = 0; i < _mParamsPtr->m; i++){
//         for (size_t j = 0; j < _mParamsPtr->n; j++){
//             float sum = 0;
//             for (size_t k = 0; k < _mParamsPtr->k; k+=_mParamsPtr->group_size){
//                 float scale = scales[(j * _mParamsPtr->k + k) / _mParamsPtr->group_size];
//                 for (size_t kk = 0; kk < _mParamsPtr->group_size; kk+=2){
//                     size_t weight_idx = (j * _mParamsPtr->k + k + kk) / 2;
//                     uint8_t weight_packed = b[weight_idx];
//                     int8_t vl = (b[weight_idx] & 0x0F) - 8;
//                     int8_t vh = (b[weight_idx] >> 4) - 8;

//                     sum += a[i * _mParamsPtr->k + k + kk] * vl * scale;
//                     sum += a[i * _mParamsPtr->k + k + kk + 1] * vh * scale;
//                 }
//             }
//             float r = result[i * _mParamsPtr->n + j];
//             if (std::abs(sum - r) / std::abs(sum) > 1e-3){
//                 std::cout << "Expect " << sum << " at " << i << "," << j << ", but getting " << r << std::endl;
//                 throw("Result verification fails!");
//             }
//         }
//     }
// }

// Fast: assuming weight format as follows
// sequential: (a, b), (c, d), (e, f), (g, h): 32 bit = 4xuint8
// expected layout of inB: (a, e), (b, f), (c, g), (d, h)
// low; (a, 0), (b, 0), (c, 0), (d, 0)
// high: (e, 0), (f, 0), (g, 0), (h, 0)
void MetalMatmulInt4::verifyResults() {
    float *a = (float *)_mBufferA->contents();
    uint8_t *b = (uint8_t *)_mBufferB->contents();
    float *result = (float *)_mBufferResult->contents();
    float *scales = (float *)_mBufferScales->contents();

    assert(_mParamsPtr->n % 2 == 0);
    for (size_t i = 0; i < _mParamsPtr->m; i++) {
        for (size_t j = 0; j < _mParamsPtr->n; j++) {
            float sum = 0;
            for (size_t k = 0; k < _mParamsPtr->k; k += _mParamsPtr->group_size) {
                float scale = scales[(j * _mParamsPtr->k + k) / _mParamsPtr->group_size];
                for (size_t kk = 0; kk < _mParamsPtr->group_size; kk += 8) {
                    size_t weight_idx = (j * _mParamsPtr->k + k + kk) / 2;
                    uint8_t weight_packed = b[weight_idx];
                    int8_t vl = (weight_packed & 0x0F) - 8;
                    int8_t vh = (weight_packed >> 4) - 8;
                    sum += a[i * _mParamsPtr->k + k + kk] * vl * scale;
                    sum += a[i * _mParamsPtr->k + k + kk + 4] * vh * scale;

                    weight_packed = b[weight_idx + 1];
                    vl = (weight_packed & 0x0F) - 8;
                    vh = (weight_packed >> 4) - 8;
                    sum += a[i * _mParamsPtr->k + k + kk + 1] * vl * scale;
                    sum += a[i * _mParamsPtr->k + k + kk + 5] * vh * scale;

                    weight_packed = b[weight_idx + 2];
                    vl = (weight_packed & 0x0F) - 8;
                    vh = (weight_packed >> 4) - 8;
                    sum += a[i * _mParamsPtr->k + k + kk + 2] * vl * scale;
                    sum += a[i * _mParamsPtr->k + k + kk + 6] * vh * scale;

                    weight_packed = b[weight_idx + 3];
                    vl = (weight_packed & 0x0F) - 8;
                    vh = (weight_packed >> 4) - 8;
                    sum += a[i * _mParamsPtr->k + k + kk + 3] * vl * scale;
                    sum += a[i * _mParamsPtr->k + k + kk + 7] * vh * scale;
                }
            }
            float r = result[i * _mParamsPtr->n + j];
            printf("%.2f, ", r);
            if (std::abs(sum - r) / std::abs(sum) > 1e-3) {
                std::cout << "Expect " << sum << " at " << i << "," << j << ", but getting " << r << std::endl;
                throw("Result verification fails!");
            }
        }
    }
}

MetalMatmulInt4::~MetalMatmulInt4() {
    _mBufferA->release();
    _mBufferB->release();
    _mBufferResult->release();

    _mMatmulFunctionPSO->release();
    _mCommandQueue->release();
}
