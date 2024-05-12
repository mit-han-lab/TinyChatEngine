// Metla logic: 
// (1) all computations are inserted as nodes;
// (2) every command buffer takes care of nodes computation by encoding nodes in correct order

#include <unordered_map>
#include <iostream>

#include <dispatch/dispatch.h>
#include <dispatch/queue.h>
#include "../include/metal_compute.h"

#define block_size 32

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

bool init_graph(int initial_capacity) {
    mgraph->mm_nodes = (const metal_params **)malloc(initial_capacity * sizeof(metal_params *));
    if (mgraph->mm_nodes == nullptr) {
        return false; // Allocation failed
    }
    mgraph->n_nodes = 0;
    mgraph->capacity = initial_capacity;
    return true;
}

void init() {
    ctx = new(struct metal_context);
    // load metal compute graph
    mgraph = new(struct metal_cgraph);
    init_graph(100);
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
        // TODO: syntax error
        METAL_ADD_KERNEL(METAL_KERNEL_FLOAT2HALF "float2half", true);
        METAL_ADD_KERNEL(METAL_KERNEL_HALF2FLOAT, "half2float", true);
        METAL_ADD_KERNEL(METAL_KERNEL_PREPARE_DECODER_ATTENTION_MASK_HALF, "kernel_prepare_decoder_attention_mask_half", true);
        METAL_ADD_KERNEL(METAL_KERNEL_SILUMUL_HALF, "SiLuMul_half", true);
        METAL_ADD_KERNEL(METAL_KERNEL_ADD_HALF, "add_half", true);
        METAL_ADD_KERNEL(METAL_KERNEL_SHAPE_QKV, "shape_qkv", true);
        METAL_ADD_KERNEL(METAL_KERNEL_UNSHAPE, "unshape", true);
        METAL_ADD_KERNEL(METAL_KERNEL_TRANSPOSE_1_2IDX, "transpose_1_2idx", true);
        METAL_ADD_KERNEL(METAL_KERNEL_CHECK_INF_HALF, "check_inf_half", true);
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

enum status metal_graph_compute(struct metal_cgraph * mg) {
    // in TinyChatEngine, inputs are operations and grouped tensors 
    MTL::ComputePassDescriptor* edesc = MTL::ComputePassDescriptor::computePassDescriptor();
    edesc->setDispatchType(MTL::DispatchTypeSerial);

    const int n_nodes  = mg->n_nodes;
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
            const struct metal_params * curr_node = mg->mm_nodes[i];
            metal_kernel_type op = curr_node->op;
            if (i == -1) {
                encoder->memoryBarrier(MTL::BarrierScopeBuffers);
                continue;
            }
            switch (op) {
                case (METAL_KERNEL_FLOAT2FLOAT):
                    MTL::Buffer *id_src0 = getBufferfromPtr((curr_node->A).data_ptr); 
                    MTL::Buffer *id_dst = getBufferfromPtr((curr_node->B).half_data_ptr); 
                    encoder->setComputePipelineState(ctx->kernels[op].pipeline);
                    encoder->setBuffer(id_src0, offs_src0, 0);
                    encoder->setBuffer(id_dst, offs_src1, 1);
                    encoder->setBytes(&curr_node->sqlen, sizeof(int), 2);
                    MTL::Size ThreadperGroup = MTL::Size::Make(1024, 1, 1);
                    MTL::Size ThreadgroupsperGrid = MTL::Size::Make((curr_node->sqlen + 1024 - 1) / 1024, 1, 1);
                    encoder->dispatchThreadgroups(ThreadgroupsperGrid, ThreadperGroup);
                    break;
                case (METAL_KERNEL_HALF2FLOAT):
                    MTL::Buffer *id_src0 = getBufferfromPtr((curr_node->A).half_data_ptr); 
                    MTL::Buffer *id_dst = getBufferfromPtr((curr_node->B).data_ptr); 
                    encoder->setComputePipelineState(ctx->kernels[op].pipeline);
                    encoder->setBuffer(id_src0, offs_src0, 0);
                    encoder->setBuffer(id_dst, offs_src1, 1);
                    encoder->setBytes(&curr_node->sqlen, sizeof(int), 2);
                    MTL::Size ThreadperGroup = MTL::Size::Make(1024, 1, 1);
                    MTL::Size ThreadgroupsperGrid = MTL::Size::Make((curr_node->sqlen + 1024 - 1) / 1024, 1, 1);
                    encoder->dispatchThreadgroups(ThreadgroupsperGrid, ThreadperGroup);
                    break;
                case (METAL_KERNEL_PREPARE_DECODER_ATTENTION_MASK_HALF):
                    MTL::Buffer *id_src0 = getBufferfromPtr((curr_node->A).half_data_ptr); 
                    encoder->setComputePipelineState(ctx->kernels[op].pipeline);
                    encoder->setBuffer(id_src0, offs_src0, 0);
                    encoder->setBytes(&curr_node->sqlen, sizeof(int), 1);
                    encoder->setBytes(&curr_node->past_sqlen, sizeof(int), 2);
                    MTL::Size ThreadperGroup = MTL::Size::Make(32, 32, 1);
                    MTL::Size ThreadgroupsperGrid = MTL::Size::Make((curr_node->sqlen - curr_node->past_sqlen + 32 - 1) / 32,
                   (curr_node->sqlen + 32 - 1) / 32, 1);
                    encoder->dispatchThreadgroups(ThreadgroupsperGrid, ThreadperGroup);
                    break;
                case (METAL_KERNEL_SILUMUL_HALF):
                    MTL::Buffer *id_src0 = getBufferfromPtr((curr_node->A).half_data_ptr); 
                    MTL::Buffer *id_src1 = getBufferfromPtr((curr_node->B).half_data_ptr);
                    encoder->setComputePipelineState(ctx->kernels[op].pipeline);
                    encoder->setBuffer(id_src0, offs_src0, 0);
                    encoder->setBuffer(id_src1, offs_src1, 1);
                    encoder->setBytes(&curr_node->sqlen, sizeof(int), 2);
                    MTL::Size ThreadperGroup = MTL::Size::Make(1024, 1, 1);
                    MTL::Size ThreadgroupsperGrid = MTL::Size::Make((curr_node->sqlen + 1024 - 1) / 1024, 1, 1);
                    encoder->dispatchThreadgroups(ThreadgroupsperGrid, ThreadperGroup);
                    break;
                case (METAL_KERNEL_ADD_HALF):
                    MTL::Buffer *id_src0 = getBufferfromPtr((curr_node->A).half_data_ptr); 
                    MTL::Buffer *id_src1 = getBufferfromPtr((curr_node->B).half_data_ptr);
                    MTL::Buffer *id_src2  = getBufferfromPtr((curr_node->C).half_data_ptr);
                    encoder->setComputePipelineState(ctx->kernels[op].pipeline);
                    encoder->setBuffer(id_src0, offs_src0, 0);
                    encoder->setBuffer(id_src1, offs_src1, 1);
                    encoder->setBuffer(id_src2,  offs_src2, 2);
                    encoder->setBytes(&curr_node->sqlen, sizeof(int), 3);
                    MTL::Size ThreadperGroup = MTL::Size::Make(1024, 1, 1);
                    MTL::Size ThreadgroupsperGrid = MTL::Size::Make((curr_node->sqlen + 1024 - 1) / 1024, 1, 1);
                    encoder->dispatchThreadgroups(ThreadgroupsperGrid, ThreadperGroup);
                    break;
                case (METAL_KERNEL_SHAPE_QKV):
                    MTL::Buffer *id_src0 = getBufferfromPtr((curr_node->A).half_data_ptr); //input_ids int
                    MTL::Buffer *id_src1 = getBufferfromPtr((curr_node->B).half_data_ptr); //output half
                    MTL::Buffer *id_src2  = getBufferfromPtr((curr_node->C).half_data_ptr);
                    MTL::Buffer *id_src3  = getBufferfromPtr((curr_node->D).half_data_ptr);
                    encoder->setComputePipelineState(ctx->kernels[op].pipeline);
                    encoder->setBuffer(id_src0, offs_src0, 0);
                    encoder->setBuffer(id_src1, offs_src1, 1);
                    encoder->setBuffer(id_src2,  offs_src2, 2);
                    encoder->setBuffer(id_src3,  offs_dst, 3);
                    encoder->setBytes(&curr_node->num_heads, sizeof(int), 4);
                    encoder->setBytes(&curr_node->sqlen, sizeof(int), 5);
                    encoder->setBytes(&curr_node->head_dim, sizeof(int), 6);
                    MTL::Size ThreadperGroup = MTL::Size::Make(16, 1, 64);
                    MTL::Size ThreadgroupsperGrid = MTL::Size::Make((curr_node->num_heads + 16 - 1) / 16,
                (curr_node->sqlen + 1 - 1) / 1y,
                (curr_node->head_dim + 64 - 1) / 64);
                    encoder->dispatchThreadgroups(ThreadgroupsperGrid, ThreadperGroup);
                    break;
                case (METAL_KERNEL_UNSHAPE):
                    MTL::Buffer *id_src0 = getBufferfromPtr((curr_node->A).half_data_ptr); 
                    MTL::Buffer *id_src1 = getBufferfromPtr((curr_node->B).half_data_ptr); 
                    encoder->setComputePipelineState(ctx->kernels[op].pipeline);
                    encoder->setBuffer(id_src0, offs_src0, 0);
                    encoder->setBuffer(id_src1, offs_src1, 1);
                    encoder->setBytes(&curr_node->num_heads, sizeof(int), 2);
                    encoder->setBytes(&curr_node->sqlen, sizeof(int), 3);
                    encoder->setBytes(&curr_node->head_dim, sizeof(int), 4);
                    MTL::Size ThreadperGroup = MTL::Size::Make(16, 1, 64);
                    MTL::Size ThreadgroupsperGrid = MTL::Size::Make((curr_node->num_heads + 16 - 1) / 16,
                (curr_node->sqlen + 1 - 1) / 1,
                (curr_node->head_dim + 64 - 1) / 64);
                    encoder->dispatchThreadgroups(ThreadgroupsperGrid, ThreadperGroup);
                    break;
                case (METAL_KERNEL_TRANSPOSE_1_2IDX):
                    MTL::Buffer *id_src0 = getBufferfromPtr((curr_node->A).half_data_ptr); 
                    MTL::Buffer *id_src1 = getBufferfromPtr((curr_node->B).half_data_ptr); 
                    encoder->setComputePipelineState(ctx->kernels[op].pipeline);
                    encoder->setBuffer(id_src0, offs_src0, 0);
                    encoder->setBuffer(id_src1, offs_src1, 1);
                    encoder->setBytes(&curr_node->A.row, sizeof(int), 2);
                    encoder->setBytes(&curr_node->A.column, sizeof(int), 3);
                    encoder->setBytes(&curr_node->input_m_dim_z, sizeof(int), 4);
                    encoder->setBytes(&curr_node->B.row, sizeof(int), 5);
                    encoder->setBytes(&curr_node->B.column, sizeof(int), 6);
                    MTL::Size ThreadperGroup = MTL::Size::Make(8, 4, 32);
                    MTL::Size ThreadgroupsperGrid = MTL::Size::Make((curr_node->num_heads + 8 - 1) / 8,
                (curr_node->tgz + 4 - 1) / 4,
                (curr_node->head_dim + 32 - 1) / 32);
                    encoder->dispatchThreadgroups(ThreadgroupsperGrid, ThreadperGroup);
                    break;
                case (METAL_KERNEL_CHECK_INF_HALF):
                    MTL::Buffer *id_src0 = getBufferfromPtr((curr_node->A).half_data_ptr);
                    encoder->setComputePipelineState(ctx->kernels[op].pipeline);
                    encoder->setBuffer(id_src0, offs_src0, 0);
                    encoder->setBytes(&curr_node->sqlen, sizeof(int), 1);
                    MTL::Size ThreadperGroup = MTL::Size::Make(1024, 1, 1);
                    MTL::Size ThreadgroupsperGrid = MTL::Size::Make((curr_node->sqlen + 1024 - 1) /1024, 1, 1);
                    encoder->dispatchThreadgroups(ThreadgroupsperGrid, ThreadperGroup);
                    break;
                case (METAL_KERNEL_EMBEDDING):
                    MTL::Buffer *id_src0 = getBufferfromPtr((curr_node->A).int32_data_ptr); //input_ids int
                    MTL::Buffer *id_dst  = getBufferfromPtr((curr_node->C).half_data_ptr); //output half
                    MTL::Buffer *id_lookup  = getBufferfromPtr((curr_node->B).data_ptr); //fp32
                    encoder->setComputePipelineState(ctx->kernels[op].pipeline);
                    encoder->setBuffer(id_src0, offs_src0, 0);
                    encoder->setBuffer(id_dst, offs_src1, 1);
                    encoder->setBuffer(id_lookup,  offs_src2, 2);
                    encoder->setBytes(&curr_node->embed_dim, sizeof(int), 3);
                    int threadsPerBlock = 1024;
                    int blocksPerGrid = (curr_node->A.column + threadsPerBlock - 1) / threadsPerBlock;
                    MTL::Size ThreadperGroup = MTL::Size::Make(threadsPerBlock, 1, 1);
                    MTL::Size ThreadgroupsperGrid = MTL::Size::Make((curr_node->A.column + threadsPerBlock - 1) / threadsPerBlock, 1, 1);
                    // Dispatch the kernel
                    encoder->dispatchThreadgroups(ThreadgroupsperGrid, ThreadperGroup);
                    break;
                struct matrix src0 = curr_node->A;
                struct matrix src1 = curr_node->B;
                struct matrix dst  = curr_node->C;
                // TODO: double check the placement of parameters
                const int64_t  ne00 = src0.row; // k
                const int64_t  ne01 = src0.column;  // n
                const int64_t  ne02 = (curr_node && curr_node->bs != 0) ? curr_node->bs : 1; // bs
                const int64_t  ne03 = 1;

                const uint64_t nb00 = sizeof(unsigned char);
                const uint64_t nb01 = nb00*ne00/block_size;
                const uint64_t nb02 = nb01*ne01;
                const uint64_t nb03 = nb02*ne02;

                const int64_t  ne10 = src1.row; // k
                const int64_t  ne11 = src1.column; // m
                const int64_t  ne12 = (curr_node && curr_node->bs != 0) ? curr_node->bs : 1; // bs
                const int64_t  ne13 = 1;

                const uint64_t nb10 = sizeof(unsigned char);
                const uint64_t nb11 = nb10*ne10;
                const uint64_t nb12 = nb11*ne11;
                const uint64_t nb13 = nb12*ne12;

                const int64_t  ne0  = dst.row;
                const int64_t  ne1  = dst.column;
                const int64_t  ne2  = (curr_node && curr_node->bs != 0) ? curr_node->bs : 1;
                const int64_t  ne3  = 1;

                const uint64_t nb0  = sizeof(unsigned char);
                const uint64_t nb1  = nb0*ne0;
                const uint64_t nb2  = nb1*ne1;
                const uint64_t nb3  = nb2*ne2;
                case METAL_KERNEL_MUL_MM_INT4_F32:
                case METAL_KERNEL_MUL_MV_INT4_F32:
                case METAL_KERNEL_MUL_MM_F32_F32:
                case METAL_KERNEL_MUL_MV_F32_F32:
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
                    MTL::Buffer *id_src0 = getBufferfromPtr(src0.fp16_data_ptr);
                    MTL::Buffer *id_src1 = getBufferfromPtr(src1.int32_data_ptr);
                    MTL::Buffer *id_dst  = getBufferfromPtr(dst.fp16_data_ptr);
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
                    MTL::Buffer *id_src0 = getBufferfromPtr(src0.half_data_ptr);
                    MTL::Buffer *id_src1 = getBufferfromPtr(src1.half_data_ptr);
                    MTL::Buffer *id_dst  = getBufferfromPtr(dst.half_data_ptr);
                    // TODO: add src1 (weights)
                    encoder->setComputePipelineState(ctx->kernels[op].pipeline);
                    encoder->setBuffer(id_src0, offs_src0, 0);
                    encoder->setBuffer(id_dst,  offs_src2, 1);
                    encoder->setBytes(&ne00, sizeof(ne00), 2);
                    encoder->setBytes(&nb01, sizeof(nb01), 3);
                    encoder->setBytes(&(curr_node->eps), sizeof(curr_node->eps), 4);
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
                    const float scale = curr_node->scale;
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
                    MTL::Buffer *id_src0 = getBufferfromPtr(src0.half_data_ptr);
                    MTL::Buffer *id_src1 = getBufferfromPtr(src1.int32_data_ptr);
                    MTL::Buffer *id_dst  = getBufferfromPtr(dst.half_data_ptr);
                    const int nth = MIN(1024, ne00);

                    const int n_past     = curr_node->n_past;     //((int32_t *) dst.op_params)[0];
                    const int n_dims     = curr_node->n_dims;     //((int32_t *) dst.op_params)[1];
                    const int mode       = curr_node->mode;       //((int32_t *) dst.op_params)[2];
                    // skip 3, n_ctx, used in GLM RoPE, unimplemented in metal
                    const int n_orig_ctx = curr_node->n_orig_ctx; //((int32_t *) dst.op_params)[4];

                    float freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow;
                    memcpy(&freq_base,   (int32_t *) curr_node->freq_base, sizeof(float)); //5
                    memcpy(&freq_scale,  (int32_t *) curr_node->freq_scale, sizeof(float)); //6
                    memcpy(&ext_factor,  (int32_t *) curr_node->ext_factor, sizeof(float)); //7
                    memcpy(&attn_factor, (int32_t *) curr_node->attn_factor, sizeof(float)); //8
                    memcpy(&beta_fast,   (int32_t *) curr_node->beta_fast, sizeof(float)); //9
                    memcpy(&beta_slow,   (int32_t *) curr_node->beta_slow, sizeof(float)); //10
                    MTL::ComputePipelineState *pipeline = ctx->kernels[METAL_KERNEL_ROPE].pipeline;
                    encoder->setComputePipelineState(ctx->kernels[op].pipeline);
                    encoder->setBuffer(id_src0, offs_src0, 0);
                    encoder->setBuffer(id_src1, offs_src1, 1);
                    encoder->setBuffer(id_dst,  offs_src2, 2);
                    encoder->setBytes(&ne00, sizeof(int64_t), 3);
                    encoder->setBytes(&ne01, sizeof(int64_t), 4);
                    encoder->setBytes(&ne02, sizeof(int64_t), 5);
                    encoder->setBytes(&ne03, sizeof(int64_t), 6);
                    encoder->setBytes(&nb00, sizeof(uint64_t), 7);
                    encoder->setBytes(&nb01, sizeof(uint64_t), 8);
                    encoder->setBytes(&nb02, sizeof(uint64_t), 9);
                    encoder->setBytes(&nb03, sizeof(uint64_t), 10);
                    encoder->setBytes(&ne0, sizeof(int64_t), 11);
                    encoder->setBytes(&ne1, sizeof(int64_t), 12);
                    encoder->setBytes(&ne2, sizeof(int64_t), 13);
                    encoder->setBytes(&ne3, sizeof(int64_t), 14);
                    encoder->setBytes(&nb0, sizeof(uint64_t), 15);
                    encoder->setBytes(&nb1, sizeof(uint64_t), 16);
                    encoder->setBytes(&nb2, sizeof(uint64_t), 17);
                    encoder->setBytes(&nb3, sizeof(uint64_t), 18);

                    encoder->setBytes(&n_past, sizeof(int), 19);
                    encoder->setBytes(&n_dims, sizeof(int), 20);
                    encoder->setBytes(&mode, sizeof(int), 21);
                    encoder->setBytes(&n_orig_ctx, sizeof(int), 22);
                    encoder->setBytes(&freq_base, sizeof(float), 23);
                    encoder->setBytes(&freq_scale, sizeof(float), 24);
                    encoder->setBytes(&ext_factor, sizeof(float), 25);
                    encoder->setBytes(&attn_factor, sizeof(float), 26);
                    encoder->setBytes(&beta_fast, sizeof(float), 27);
                    encoder->setBytes(&beta_slow, sizeof(float), 28);

                    MTL::Size ThreadperGroup = MTL::Size::Make(nth, 1, 1);
                    MTL::Size ThreadgroupsperGrid = MTL::Size::Make(ne01, ne02, ne03); // from https://github.com/ggerganov/llama.cpp/blob/1b496a745c315022df2d919374052e6004ced8d3/ggml-metal.m#L2240
                    // Dispatch the kernel
                    encoder->dispatchThreadgroups(ThreadgroupsperGrid, ThreadperGroup);
                    break;
            }
            if (encoder!=nullptr){
                encoder->endEncoding();
                encoder=nullptr;
            }
        }
        command_buffer->commit();
        command_buffer->waitUntilCompleted();
        if (command_buffer->status()!=MTL::CommandBufferStatusCompleted){
            return STATUS_FAILED;
        }
    }
    return STATUS_SUCCESS;
}

void add_node(const struct metal_params *new_node) {
    if (mgraph == nullptr) {
        std::cerr << "Graph is null" << std::endl;
        return;
    }
    if (mgraph->n_nodes == mgraph->capacity) {
        // Need more space, so let's double the capacity
        int new_capacity = mgraph->capacity * 2;
        const metal_params **new_mm_nodes = (const metal_params **)realloc(mgraph->mm_nodes, new_capacity * sizeof(metal_params *));
        if (new_mm_nodes == nullptr) {
            std::cerr << "Memory allocation failed" << std::endl;
            return; // Keep the old memory intact
        }
        mgraph->mm_nodes = new_mm_nodes;
        mgraph->capacity = new_capacity;
    }

    // Add the new node at the end of the array
    mgraph->mm_nodes[mgraph->n_nodes] = new_node;
    mgraph->n_nodes++;
}