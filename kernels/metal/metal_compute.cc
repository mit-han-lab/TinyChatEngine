#include <unordered_map>

#include "Foundation/Foundation.hpp"
#include "Metal/Metal.hpp"
#include <dispatch/dispatch.h>

#undef MIN
#undef MAX
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))


bool has_init = false;

struct ggml_metal_kernel {
    MTL::ComputePipelineState * pipeline;
};

struct ggml_metal_context * ctx;

enum {
    MTLGPUFamilyApple1 = 1001, // Example starting value, adjust based on actual definition
    MTLGPUFamilyCommon1 = 3001, // Example starting value
    MTLGPUFamilyMetal3 = 5001,
    MTLGPUFamilyApple7 = 1007,
};

enum ggml_metal_kernel_type {
    GGML_METAL_KERNEL_EMBEDDING,
    GGML_METAL_KERNEL_BATCH_ADD,
    GGML_METAL_KERNEL_RELU,
    GGML_METAL_KERNEL_SILU,
    GGML_METAL_KERNEL_GELU,
    GGML_METAL_KERNEL_GELU_QUICK,
    GGML_METAL_KERNEL_RMS_NORM,
    GGML_METAL_KERNEL_SOFT_MAX,
    GGML_METAL_KERNEL_SOFT_MAX_4,
    GGML_METAL_KERNEL_ROPE,
    GGML_METAL_KERNEL_MUL_MM_INT4,
    GGML_METAL_KERNEL_TYPE_COUNT
};

enum ggml_status {
    GGML_STATUS_SUCCESS,
    GGML_STATUS_FAILED
};

// Context struct holding Metal related objects and state
struct ggml_metal_context {
    int n_cb;

    MTL::Device * device;
    MTL::CommandQueue * queue;
    static std::unordered_map<void *, MTL::Buffer *> _mumap;

    dispatch_queue_t d_queue;

    ggml_metal_kernel kernels[GGML_METAL_KERNEL_TYPE_COUNT];

    bool support_simdgroup_reduction;
    bool support_simdgroup_mm;

    bool should_capture_next_compute;
};
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
    ctx = new(struct ggml_metal_context);
    MTL::Device  *device = MTL::CreateSystemDefaultDevice();
    ctx->device = device;
    ctx->n_cb   = 1; // TODO: n_cb and GGML_METAL_MAX_BUFFERS? MIN(n_cb, GGML_METAL_MAX_BUFFERS)
    ctx->queue = ctx->device->newCommandQueue();
    ctx->d_queue = dispatch_queue_create("ggml-metal", DISPATCH_QUEUE_CONCURRENT);
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
        for (int i = 0; i < GGML_METAL_KERNEL_TYPE_COUNT; ++i) {
            ctx->kernels[i].pipeline = nullptr;
        }
#define GGML_METAL_ADD_KERNEL(e, name, supported) \
        if (supported) { \
            struct ggml_metal_kernel * kernel = &ctx->kernels[e]; \
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
        GGML_METAL_ADD_KERNEL(GGML_METAL_KERNEL_EMBEDDING, "embedding", true);
        GGML_METAL_ADD_KERNEL(GGML_METAL_KERNEL_BATCH_ADD, "batch_add", true);
        GGML_METAL_ADD_KERNEL(GGML_METAL_KERNEL_RELU, "relu", true);
        GGML_METAL_ADD_KERNEL(GGML_METAL_KERNEL_SILU, "silu", true);
        GGML_METAL_ADD_KERNEL(GGML_METAL_KERNEL_GELU, "gelu", true);
        GGML_METAL_ADD_KERNEL(GGML_METAL_KERNEL_GELU_QUICK, "gelu_quick", true);
        GGML_METAL_ADD_KERNEL(GGML_METAL_KERNEL_RMS_NORM, "rms_norm", true);
        GGML_METAL_ADD_KERNEL(GGML_METAL_KERNEL_SOFT_MAX, "soft_max", true);
        GGML_METAL_ADD_KERNEL(GGML_METAL_KERNEL_SOFT_MAX_4, "soft_max_4", true);
        GGML_METAL_ADD_KERNEL(GGML_METAL_KERNEL_ROPE, "rope", true);
        GGML_METAL_ADD_KERNEL(GGML_METAL_KERNEL_MUL_MM_INT4, "mul_mm_int4", true);        
    }
    metal_library->release();
    has_init = true;
}

MTL::Buffer *MetalIMP::getBufferfromPtr(void *ptr) {
    if (_mumap.find(ptr) == _mumap.end()) {
        std::cerr << "Cannot find the corresponding MTL::Buffer." << std::endl;
        return NULL;
    } else
        return _mumap[ptr];
}

static void ggml_metal_free(struct ggml_metal_context * ctx) {
    for (int i = 0; i < GGML_METAL_KERNEL_TYPE_COUNT; ++i) {
        ctx->kernels[i].pipeline->release();
    }

    ctx->queue->release();
    ctx->device->release();

    dispatch_release(ctx->d_queue);

    free(ctx);
}

static enum ggml_status ggml_metal_graph_compute(struct ggml_metal_context * ctx,
               struct ggml_cgraph * gf) {

    return GGML_STATUS_SUCCESS;
}