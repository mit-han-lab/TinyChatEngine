import Metal

let metalSourceCode = """
    kernel void arrayAdd(const device float* inputA [[buffer(0)]],
                          const device float* inputB [[buffer(1)]],
                          device float* output [[buffer(2)]],
                          uint id [[thread_position_in_grid]])
    {
        // Perform array addition
        output[id] = inputA[id] + inputB[id];
    }
"""
let arraySize: Int = 1*108*768 // simulate the Matrix3D size in test_ops.cc
let inputA = getRandomArray()
let inputB = getRandomArray()
let output = [Float](repeating: 1.0, count: arraySize)

var timeElapsedGPU: CFAbsoluteTime = 0.0
var timeElapsedCPU: CFAbsoluteTime = 0.0


func performArrayAddition() {
    print("Metal GPU")
    // Begin the process
    let startTime = CFAbsoluteTimeGetCurrent()

    // Create Metal device and command queue
    let device = MTLCreateSystemDefaultDevice()!
    let commandQueue = device.makeCommandQueue()!

    // Create Metal buffers for input and output
    let bufferA = device.makeBuffer(bytes: inputA, length: arraySize * MemoryLayout<Float>.size, options: .storageModeShared)!
    let bufferB = device.makeBuffer(bytes: inputB, length: arraySize * MemoryLayout<Float>.size, options: .storageModeShared)!
    let bufferOutput = device.makeBuffer(bytes: output, length: arraySize * MemoryLayout<Float>.size, options: .storageModeShared)!

    // Create Metal compute pipeline and set kernel function
    let library = try! device.makeLibrary(source: metalSourceCode, options: nil)
    let kernelFunction = library.makeFunction(name: "arrayAdd")!
    let pipeline = try! device.makeComputePipelineState(function: kernelFunction)

    // Create Metal compute command encoder
    let commandBuffer = commandQueue.makeCommandBuffer()!
    let computeEncoder = commandBuffer.makeComputeCommandEncoder()!

    // Set compute pipeline state
    computeEncoder.setComputePipelineState(pipeline)

    // Set buffers for input and output
    computeEncoder.setBuffer(bufferA, offset: 0, index: 0)
    computeEncoder.setBuffer(bufferB, offset: 0, index: 1)
    computeEncoder.setBuffer(bufferOutput, offset: 0, index: 2)

    // Set threadgroup size and dispatch compute threads
    let maxThreadsperthreadgroup = pipeline.maxTotalThreadsPerThreadgroup

    let threadsPerThreadgroup = min(arraySize, maxThreadsperthreadgroup)
    let threadgroupCount = MTLSize(width: (arraySize + threadsPerThreadgroup - 1) / threadsPerThreadgroup, height: 1, depth: 1)

    // Dispatch threads in multiple threadgroups
    let threadgroups = MTLSize(width: threadsPerThreadgroup, height: 1, depth: 1)
    computeEncoder.dispatchThreads(threadgroups, threadsPerThreadgroup: threadgroupCount)

    // End encoding and execute command buffer
    computeEncoder.endEncoding()
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()

    // Access the result from the output buffer
    var resultBufferPointer = UnsafeMutablePointer<Float>(bufferOutput.contents().bindMemory(to: Float.self,
                                                                capacity: MemoryLayout<Float>.size * arraySize))
    
    // Print the result
    for i in 0..<3 {
        print("Metal GPU result: \(inputA[i]) + \(inputB[i]) = \(Float(resultBufferPointer.pointee) as Any)")
        resultBufferPointer = UnsafeMutablePointer<Float>(resultBufferPointer.advanced(by: 1))
    }

    timeElapsedGPU = CFAbsoluteTimeGetCurrent() - startTime
    print("Time elapsed \(String(format: "%.05f", timeElapsedGPU)) seconds")
    print()
}

// Call the function to perform array addition using Metal GPU
performArrayAddition()
basicForLoopWay(arr1: inputA, arr2: inputB)
let speedup = timeElapsedCPU/timeElapsedGPU
print("Speedup: \(speedup)")

func basicForLoopWay(arr1: [Float], arr2: [Float]) {
    print("CPU")
    
    // Begin the process
    let startTime = CFAbsoluteTimeGetCurrent()

    var result = [Float].init(repeating: 0.0, count: arraySize)

    // Process our additions of the arrays together
    for i in 0..<arraySize {
        result[i] = arr1[i] + arr2[i]
    }

    // Print out the results
    for i in 0..<3 {
        print("CPU result: \(arr1[i]) + \(arr2[i]) = \(result[i])")
    }

    // Print out the elapsed time
    timeElapsedCPU = CFAbsoluteTimeGetCurrent() - startTime
    print("Time elapsed \(String(format: "%.05f", timeElapsedCPU)) seconds")

    print()
}

func getRandomArray()->[Float] {
    var result = [Float].init(repeating: 0.0, count: arraySize)
    for i in 0..<arraySize {
        result[i] = Float(arc4random_uniform(10))
    }
    return result
}