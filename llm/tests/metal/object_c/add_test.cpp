#include <iostream>
#include <Metal/Metal.h>


class MetalArrayAddition {
public:
    MetalArrayAddition() {
        // Initialize Metal
        device = MTLCreateSystemDefaultDevice();
        if (!device) {
            std::cerr << "Metal is not supported on this device." << std::endl;
            exit(EXIT_FAILURE);
        }

        // Create a command queue
        commandQueue = [device newCommandQueue];
        
        // Load and compile the Metal Shaders
        NSError *error = nil;
        NSString *shaderSource = [NSString stringWithContentsOfFile:@"add.metal"
                                                         encoding:NSUTF8StringEncoding
                                                            error:&error];
        if (error) {
            std::cerr << "Error reading shader source: " << error.localizedDescription.UTF8String << std::endl;
            exit(EXIT_FAILURE);
        }
        
        NSBundle *bundle = [NSBundle mainBundle];
        NSString *shaderPath = [bundle pathForResource:@"ArrayAdditionShader" ofType:@"metal"];
        NSURL *shaderURL = [NSURL fileURLWithPath:shaderPath];
        
        NSError *compileError = nil;
        library = [device newLibraryWithSource:shaderSource options:nil error:&compileError];
        if (compileError) {
            std::cerr << "Shader compilation error: " << compileError.localizedDescription.UTF8String << std::endl;
            exit(EXIT_FAILURE);
        }

        // Create a pipeline state
        MTLFunction *kernelFunction = [library newFunctionWithName:@"elementwise_add"];
        pipelineState = [device newComputePipelineStateWithFunction:kernelFunction error:&error];
        if (error) {
            std::cerr << "Pipeline state creation error: " << error.localizedDescription.UTF8String << std::endl;
            exit(EXIT_FAILURE);
        }
    }

    void performArrayAddition(const float* inputArrayA, const float* inputArrayB, float* outputArray, int arraySize) {
        // Create buffers
        MTLBuffer *bufferA = [device newBufferWithBytes:inputArrayA length:arraySize * sizeof(float) options:MTLResourceStorageModeShared];
        MTLBuffer *bufferB = [device newBufferWithBytes:inputArrayB length:arraySize * sizeof(float) options:MTLResourceStorageModeShared];
        MTLBuffer *bufferResult = [device newBufferWithLength:arraySize * sizeof(float) options:MTLResourceStorageModeShared];

        // Create a command buffer
        id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];

        // Create a compute command encoder
        id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
        [computeEncoder setComputePipelineState:pipelineState];
        [computeEncoder setBuffer:bufferA offset:0 atIndex:0];
        [computeEncoder setBuffer:bufferB offset:0 atIndex:1];
        [computeEncoder setBuffer:bufferResult offset:0 atIndex:2];

        // Set thread group size
        MTLSize threadGroupSize = MTLSizeMake(1, 1, 1);
        MTLSize threadGroups = MTLSizeMake(arraySize / threadGroupSize.width, 1, 1);

        // Dispatch the compute kernel
        [computeEncoder dispatchThreadgroups:threadGroups threadsPerThreadgroup:threadGroupSize];
        [computeEncoder endEncoding];

        // Commit the command buffer
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];

        // Read back the result
        memcpy(outputArray, [bufferResult contents], arraySize * sizeof(float));
    }

private:
    id<MTLDevice> device;
    id<MTLCommandQueue> commandQueue;
    id<MTLLibrary> library;
    id<MTLComputePipelineState> pipelineState;
};

int main() {
    const int arraySize = 100;
    float inputArrayA[arraySize];
    float inputArrayB[arraySize];
    float outputArray[arraySize];

    // Initialize input arrays
    for (int i = 0; i < arraySize; ++i) {
        inputArrayA[i] = i;
        inputArrayB[i] = 2 * i;
    }

    // Create MetalArrayAddition instance
    MetalArrayAddition metalArrayAddition;

    // Perform array addition using Metal GPU
    metalArrayAddition.performArrayAddition(inputArrayA, inputArrayB, outputArray, arraySize);

    // Display the result
    std::cout << "Resultant Array:" << std::endl;
    for (int i = 0; i < arraySize; ++i) {
        std::cout << outputArray[i] << " ";
    }

    return 0;
}
