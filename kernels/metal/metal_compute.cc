#include <unordered_map>
#include <iostream>

#include <dispatch/queue.h>
#include "metal_compute.h"

#undef MIN
#undef MAX
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

void *allocateSharedMem(size_t size) {
    if (!has_init) {
        init();
    }

    MTL::Buffer *new_b = ctx->device->newBuffer(size, MTL::ResourceStorageModeShared);

    void *void_ptr = new_b->contents();

    // push the pair to the map
    ctx->_mumap.insert(std::make_pair(void_ptr, new_b));

    return void_ptr;
}

void init() {
    ctx = new(struct metal_context);
    MTL::Device  *device = MTL::CreateSystemDefaultDevice();
    ctx->device = device;
    ctx->n_cb   = 1; // TODO: n_cb and METAL_MAX_BUFFERS? MIN(n_cb, METAL_MAX_BUFFERS=64)
    ctx->queue = ctx->device->newCommandQueue();
    // ctx->d_queue = dispatch_queue_create("ggml-metal", DISPATCH_QUEUE_CONCURRENT);
    MTL::Library *metal_library = ctx->device->newDefaultLibrary();

    // simd group support
    for (int i = MTLGPUFamilyApple1 + 20; i >= MTLGPUFamilyApple1; --i) {
        if (ctx->device->supportsFamily((MTL::GPUFamily)i)) {
            printf("%s: GPU family: MTLGPUFamilyApple%d  (%d)\n", __FUNCTION__, i - MTLGPUFamilyApple1 + 1, i);
            break;
        }
    }

    for (int i = MTLGPUFamilyCommon1 + 5; i >= MTLGPUFamilyCommon1; --i) {
        if (ctx->device->supportsFamily((MTL::GPUFamily)i)) {
            printf("%s: GPU family: MTLGPUFamilyCommon%d (%d)\n", __FUNCTION__, i - MTLGPUFamilyCommon1 + 1, i);
            break;
        }
    }

    for (int i = MTLGPUFamilyMetal3 + 5; i >= MTLGPUFamilyMetal3; --i) {
        if (ctx->device->supportsFamily((MTL::GPUFamily)i)) {
            printf("%s: GPU family: MTLGPUFamilyMetal%d  (%d)\n", __FUNCTION__, i - MTLGPUFamilyMetal3 + 3, i);
            break;
        }
    }

    ctx->support_simdgroup_reduction = ctx->device->supportsFamily((MTL::GPUFamily)MTLGPUFamilyApple7);
    ctx->support_simdgroup_reduction |= ctx->device->supportsFamily((MTL::GPUFamily)MTLGPUFamilyMetal3);
    ctx->support_simdgroup_mm = ctx->device->supportsFamily((MTL::GPUFamily)MTLGPUFamilyApple7);


    // load kernels
    {
        NS::Error *error = nullptr;
        for (int i = 0; i < METAL_KERNEL_TYPE_COUNT; ++i) {
            ctx->kernels[i].pipeline = nullptr;
        }
#define METAL_ADD_KERNEL(e, name, supported) \
        if (supported) { \
            struct metal_kernel * kernel = &ctx->kernels[e]; \
            const char * str = "kernel_" + name; \
            auto str = NS::String::string(str, NS::ASCIIStringEncoding); \
            MTL::Function * metal_function = metal_library->newFunction(str); \
            kernel->pipeline  = ctx->device->newComputePipelineState(metal_function, &error); \
            metal_function->release(); \
            if (error) { \
                printf("load pipeline error"); \
                return nullptr; \
            } \    
        } else { \
            printf("kernel name not supported "); \
        }

        // simd_sum and simd_max requires MTLGPUFamilyApple7
        // TODO: solve error
        METAL_ADD_KERNEL(METAL_KERNEL_EMBEDDING, "embedding", true);
        METAL_ADD_KERNEL(METAL_KERNEL_BATCH_ADD, "batch_add", true);
        METAL_ADD_KERNEL(METAL_KERNEL_RELU, "relu", true);
        METAL_ADD_KERNEL(METAL_KERNEL_SILU, "silu", true);
        METAL_ADD_KERNEL(METAL_KERNEL_GELU, "gelu", true);
        METAL_ADD_KERNEL(METAL_KERNEL_GELU_QUICK, "gelu_quick", true);
        METAL_ADD_KERNEL(METAL_KERNEL_RMS_NORM, "rms_norm", true);
        METAL_ADD_KERNEL(METAL_KERNEL_SOFT_MAX, "soft_max", true);
        METAL_ADD_KERNEL(METAL_KERNEL_SOFT_MAX_4, "soft_max_4", true);
        METAL_ADD_KERNEL(METAL_KERNEL_ROPE, "rope", true);
        METAL_ADD_KERNEL(METAL_KERNEL_MUL_MM_INT4_F32, "mul_mm_int4_f32", true);
        METAL_ADD_KERNEL(METAL_KERNEL_MUL_MV_INT4_F32, "mul_mv_int4_f32", true);  
        METAL_ADD_KERNEL(METAL_KERNEL_MUL_MM_F32_F32, "mul_mm_f32_f32", true);  
        METAL_ADD_KERNEL(METAL_KERNEL_MUL_MV_F32_F32, "mul_mv_f32_f32", true);          
    }
    metal_library->release();
    has_init = true;
}

MTL::Buffer *getBufferfromPtr(void *ptr) {
    if (ctx->_mumap.find(ptr) == ctx->_mumap.end()) {
        std::cerr << "Cannot find the corresponding MTL::Buffer." << std::endl;
        return NULL;
    } else
        return ctx->_mumap[ptr];
}

static void metal_free(struct metal_context * ctx) {
    for (int i = 0; i < METAL_KERNEL_TYPE_COUNT; ++i) {
        ctx->kernels[i].pipeline->release();
    }

    ctx->queue->release();
    ctx->device->release();

    // dispatch_release(ctx->d_queue);

    free(ctx);
}

static enum status metal_graph_compute(metal_kernel_type op, 
    struct metal_cgraph * metal_data) {
    // in TinyChatEngine, inputs are operations and grouped tensors 
    MTL::ComputePassDescriptor* edesc = MTL::ComputePassDescriptor::computePassDescriptor();
    edesc->setDispatchType(MTL::DispatchTypeSerial);

    const int n_nodes  = metal_data->n_nodes;
    const int n_cb = ctx->n_cb; // number of command buffer, TODO: currently 1 in TinyChatEngine
    const int n_nodes_per_cb = (n_nodes + n_cb - 1) / n_cb;

    MTL::CommandBuffer *command_buffer_builder[n_cb];
    for (int cb_idx = 0; cb_idx < n_cb; ++cb_idx) {
        MTL::CommandBuffer *command_buffer  = ctx->queue->commandBufferWithUnretainedReferences();
        command_buffer_builder[cb_idx] = command_buffer;
        // enqueue the command buffers in order to specify their execution order
        command_buffer->enqueue();
    }
    MTL::CommandBuffer **command_buffers = command_buffer_builder;
    for (int iter = 0; iter < n_cb; ++iter){
        const int cb_idx = iter;
        size_t offs_src0 = 0;
        size_t offs_src1 = 0;
        size_t offs_src2 = 0;
        size_t offs_dst  = 0;
        MTL::CommandBuffer *command_buffer  = command_buffers[cb_idx];
        MTL::ComputeCommandEncoder *encoder = command_buffer->computeCommandEncoder(edesc);

        const int node_start =                                      (cb_idx + 0) * n_nodes_per_cb;
        const int node_end   = MIN((cb_idx == n_cb - 1) ? n_nodes : (cb_idx + 1) * n_nodes_per_cb, n_nodes);

        for (int i = node_start; i < node_end; ++i) {
            if (i == -1) {
                encoder->memoryBarrier(MTL::BarrierScopeBuffers);
                continue;
            }
            switch (op) {
                case (METAL_KERNEL_EMBEDDING):
                    MTL::Buffer *id_src0 = getBufferfromPtr(metal_data->input_id);
                    MTL::Buffer *id_dst  = getBufferfromPtr(metal_data->output);
                    MTL::Buffer *id_lookup  = getBufferfromPtr(metal_data->lookup);
                    encoder->setComputePipelineState(ctx->kernels[op].pipeline);
                    encoder->setBuffer(id_src0, offs_src0, 0);
                    encoder->setBuffer(id_dst, offs_src1, 1);
                    encoder->setBuffer(id_lookup,  offs_src2, 2);
                    encoder->setBytes(&metal_data->op_constants.embed_dim, sizeof(embed_dim), 3);
                    int threadsPerBlock = 1024;
                    int blocksPerGrid = (metal_data->input_id.m_dim_z + threadsPerBlock - 1) / threadsPerBlock;
                    MTL::Size ThreadperGroup = MTL::Size::Make(threadsPerBlock, 1, 1);
                    MTL::Size ThreadgroupsperGrid = MTL::Size::Make((input_id.m_dim_z + threadsPerBlock - 1) / threadsPerBlock, 1, 1);
                    // Dispatch the kernel
                    encoder->dispatchThreadgroups(ThreadgroupsperGrid, ThreadperGroup);
                    break;
                const metal_params * inputs = metal_data->mm_nodes[i];
                struct matrix src0 = inputs->A;
                struct matrix src1 = inputs->B;
                struct matrix dst  = inputs->C;
                // TODO: ne[0], nb[0] calculation & order
                const int64_t  ne00 = src0.row;
                const int64_t  ne01 = src0.column;
                const int64_t  ne02 = 1;
                const int64_t  ne03 = 1;

                const uint64_t nb00 = (op == METAL_KERNEL_MUL_MM_INT4_F32) || (op == METAL_KERNEL_MUL_MV_INT4_F32) ? sizeof(uint8_t) : sizeof(float);
                const uint64_t nb01 = nb00*ne00;
                const uint64_t nb02 = nb01*ne01;
                const uint64_t nb03 = nb02*ne02;

                const int64_t  ne10 = src1.row;
                const int64_t  ne11 = src1.column;
                const int64_t  ne12 = 1;
                const int64_t  ne13 = 1;

                const uint64_t nb10 = sizeof(float);
                const uint64_t nb11 = nb10*ne10;
                const uint64_t nb12 = nb11*ne11;
                const uint64_t nb13 = nb12*ne12;

                const int64_t  ne0  = dst.row;
                const int64_t  ne1  = dst.column;
                const int64_t  ne2  = 1;
                const int64_t  ne3  = 1;

                const uint64_t nb0  = sizeof(float);
                const uint64_t nb1  = nb0*ne0;
                const uint64_t nb2  = nb1*ne1;
                const uint64_t nb3  = nb2*ne2;
                case METAL_KERNEL_MUL_MM_INT4_F32:
                case METAL_KERNEL_MUL_MV_INT4_F32:
                case METAL_KERNEL_MUL_MM_F32_F32:
                case METAL_KERNEL_MUL_MV_F32_F32:
                    // TODO: buffer retrieved?
                    MTL::Buffer *id_src0 = (op == METAL_KERNEL_MUL_MM_INT4_F32) || (op == METAL_KERNEL_MUL_MV_INT4_F32) ? getBufferfromPtr(src0.int4_data_ptr) : getBufferfromPtr(src0.data_ptr);
                    MTL::Buffer *id_src1 = getBufferfromPtr(src1.data_ptr);
                    MTL::Buffer *id_dst  = getBufferfromPtr(dst.data_ptr);
                    const uint r2 = ne12/ne02;
                    const uint r3 = ne13/ne03;
                    int ne11_mm_min = 1;
                    // src0 quantized; src1 F32
                    // for now the matrix-matrix multiplication kernel only works on A14+/M1+ SoCs
                    // AMD GPU and older A-chips will reuse matrix-vector multiplication kernel
                    if (ctx->device->supportsFamily((MTL::GPUFamily)MTLGPUFamilyApple7)&&
                        ne00 % 32 == 0 && ne00 >= 64){
                        encoder->setComputePipelineState(ctx->kernels[op].pipeline);
                        encoder->setBuffer(id_src0, offs_src0, 0);
                        encoder->setBuffer(id_src1, offs_src1, 1);
                        encoder->setBuffer(id_dst,  offs_src2, 2);
                        encoder->setBytes(&ne00, sizeof(ne00), 3);
                        encoder->setBytes(&ne02, sizeof(ne02), 4);
                        encoder->setBytes(&nb01, sizeof(nb01), 5);
                        encoder->setBytes(&nb02, sizeof(nb02), 6);
                        encoder->setBytes(&ne12, sizeof(ne12), 7);
                        encoder->setBytes(&nb10, sizeof(nb10), 8);
                        encoder->setBytes(&nb11, sizeof(nb11), 9);
                        encoder->setBytes(&nb12, sizeof(nb12), 10);
                        encoder->setBytes(&ne0, sizeof(ne0), 11);
                        encoder->setBytes(&ne1, sizeof(ne1), 12);
                        encoder->setBytes(&r2, sizeof(r2), 13);
                        encoder->setBytes(&r3, sizeof(r3), 14);
                        encoder->setThreadgroupMemoryLength(8192, 0);
                        MTL::Size ThreadperGroup = MTL::Size::Make(128, 1, 1);
                        MTL::Size ThreadgroupsperGrid = MTL::Size::Make((ne11 + 31)/32, (ne01 + 63)/64, ne12*ne13); // from https://github.com/ggerganov/llama.cpp/blob/d5ab29757ebc59a30f03e408294ec20628a6374e/ggml-metal.m#L1405
                        // Dispatch the kernel
                        encoder->dispatchThreadgroups(ThreadgroupsperGrid, ThreadperGroup);
                    }
                    break;
                case (METAL_KERNEL_BATCH_ADD):
                    MTL::Buffer *id_src0 = getBufferfromPtr(src0.data_ptr);
                    MTL::Buffer *id_src1 = getBufferfromPtr(src1.data_ptr);
                    MTL::Buffer *id_dst  = getBufferfromPtr(dst.data_ptr);
                    encoder->setComputePipelineState(ctx->kernels[op].pipeline);
                    encoder->setBuffer(id_src0, offs_src0, 0);
                    encoder->setBuffer(id_src1, offs_src1, 1);
                    encoder->setBuffer(id_dst,  offs_src2, 2);
                    MTL::Size ThreadperGroup = MTL::Size::Make(src0.row, src0.column, 1);
                    MTL::Size ThreadgroupsperGrid = MTL::Size::Make(1, 1, 1); 
                    encoder->dispatchThreadgroups(ThreadgroupsperGrid, ThreadperGroup);
                    break;
                case (METAL_KERNEL_RELU):
                case (METAL_KERNEL_SILU):
                case (METAL_KERNEL_GELU):
                case (METAL_KERNEL_GELU_QUICK):
                    MTL::Buffer *id_src0 = getBufferfromPtr(src0.data_ptr);
                    MTL::Buffer *id_dst  = getBufferfromPtr(dst.data_ptr);
                    encoder->setComputePipelineState(ctx->kernels[op].pipeline);
                    encoder->setBuffer(id_src0, offs_src0, 0);
                    encoder->setBuffer(id_dst,  offs_src2, 1);
                    MTL::Size ThreadperGroup = MTL::Size::Make(src0.length,1, 1);
                    MTL::Size ThreadgroupsperGrid = MTL::Size::Make(1, 1, 1); 
                    encoder->dispatchThreadgroups(ThreadgroupsperGrid, ThreadperGroup);
                    break;
                case (METAL_KERNEL_RMS_NORM):
                    int nth = 32; // SIMD width
                    const int64_t  ne00 = src0.row;
                    const int64_t  ne01 = src0.column;
                    const int64_t  ne02 = 1;
                    const int64_t  ne03 = 1;
                    // TODO: nb00 should be half?
                    const uint64_t nb00 = (op == METAL_KERNEL_MUL_MM_INT4_F32) || (op == METAL_KERNEL_MUL_MV_INT4_F32) ? sizeof(uint8_t) : sizeof(float);
                    const uint64_t nb01 = nb00*ne00;
                    const uint64_t nb02 = nb01*ne01;
                    const uint64_t nb03 = nb02*ne02;
                    MTL::Buffer *id_src0 = getBufferfromPtr(src0.half_data_ptr);
                    MTL::Buffer *id_src1 = getBufferfromPtr(src1.half_data_ptr);
                    MTL::Buffer *id_dst  = getBufferfromPtr(dst.half_data_ptr);
                    // TODO: add src1
                    encoder->setComputePipelineState(ctx->kernels[op].pipeline);
                    encoder->setBuffer(id_src0, offs_src0, 0);
                    encoder->setBuffer(id_dst,  offs_src2, 1);
                    encoder->setBytes(&ne00, sizeof(ne00), 2);
                    encoder->setBytes(&nb01, sizeof(nb01), 3);
                    encoder->setBytes(&(metal_data->op_constants.eps), sizeof(metal_data->op_constants.eps), 4);
                    encoder->setThreadgroupMemoryLength(32*sizeof(float), 0); 
                    encoder->dispatchThreadgroups(MTL::Size::Make(src0.row, 1, 1), MTL::Size::Make(src0.row, 1, 1));
                    break;
                case (METAL_KERNEL_SOFT_MAX):
                case (METAL_KERNEL_SOFT_MAX_4):
                    int nth = 32; // SIMD width
                    if (ne00%4 == 0) {
                        while (nth < ne00/4 && nth < 256) {
                            nth *= 2;
                        }
                        encoder->setComputePipelineState(ctx->kernels[op].pipeline);
                    } else {
                        while (nth < ne00 && nth < 1024) {
                            nth *= 2;
                        }
                        encoder->setComputePipelineState(ctx->kernels[op].pipeline);
                    }
                    // TODO: type
                    const int64_t  ne00 = src0.row;
                    const int64_t  ne01 = src0.column;
                    const int64_t  ne02 = 1;
                    const int64_t  ne03 = 1;
                    const float scale = metal_data->op_constants.scale;
                    MTL::Buffer *id_src0 = getBufferfromPtr(src0.half_data_ptr);
                    MTL::Buffer *id_src1 = getBufferfromPtr(src1.data_ptr);
                    MTL::Buffer *id_dst  = getBufferfromPtr(dst.data_ptr);
                    encoder->setBuffer(id_src0, offs_src0, 0);
                    encoder->setBuffer(id_src1, offs_src1, 1);
                    encoder->setBuffer(id_dst,  offs_src2, 2);
                    encoder->setBytes(&ne00, sizeof(ne00), 3);
                    encoder->setBytes(&ne01, sizeof(ne01), 4);
                    encoder->setBytes(&ne02, sizeof(ne02), 5);
                    encoder->setBytes(&scale, sizeof(scale), 6);
                    encoder->setThreadgroupMemoryLength(32*sizeof(float), 0); 
                    encoder->dispatchThreadgroups(MTL::Size::Make(ne01*ne02*ne03, 1, 1), MTL::Size::Make(nth, 1, 1));
                    break;
                case (METAL_KERNEL_ROPE):
                    //TODO: implement ROPE
                    break;
            }
            if (encoder!=nullptr){
                encoder->endEncoding();
                encoder=nullptr;
            }
            command_buffer->commit();
            command_buffer->waitUntilCompleted();
            if (command_buffer->status()!=MTL::CommandBufferStatusCompleted){
                return STATUS_FAILED;
            }
        }
    }
    return STATUS_SUCCESS;
}