#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "../../../kernels/matmul.h"

// ============================================================
// CUDA 错误检查
// ============================================================
#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err = (call);                                            \
        if (err != cudaSuccess) {                                            \
            fprintf(stderr,                                                  \
                    "CUDA error at %s:%d: %s\n",                             \
                    __FILE__, __LINE__, cudaGetErrorString(err));             \
            std::exit(EXIT_FAILURE);                                         \
        }                                                                    \
    } while (0)


// ============================================================
// TinyChatEngine 中的 make_divisible：
// ceil(c / divisor)
// ============================================================
static inline int make_divisible_host(int c, int divisor) {
    return (c + divisor - 1) / divisor;
}


// ============================================================
// 单个 GEMV 测试
//
// 输入:
//   A: [1, IC] FP16
//
// 权重:
//   B: [OC, IC / 8]
//   每个 uint32_t 存 8 个 INT4
//
// 输出:
//   C: [1, OC] FP16
//
// group_size = 128
// ============================================================
void benchmark_gemv(
    int IC,
    int OC,
    int warmup_iters = 20,
    int benchmark_iters = 200
) {
    constexpr int GROUP_SIZE = 128;
    constexpr int PACK_FACTOR = 8;

    // --------------------------------------------------------
    // 基本约束
    // --------------------------------------------------------
    if (IC % GROUP_SIZE != 0) {
        std::cerr << "IC 必须能被 128 整除\n";
        std::exit(EXIT_FAILURE);
    }

    if (IC % PACK_FACTOR != 0) {
        std::cerr << "IC 必须能被 8 整除\n";
        std::exit(EXIT_FAILURE);
    }

    if (OC % 4 != 0) {
        std::cerr << "OC 必须能被 4 整除\n";
        std::exit(EXIT_FAILURE);
    }

    // ========================================================
    // 1. 计算各种数据的尺寸
    // ========================================================

    // 输入 FP16 元素数量
    const size_t input_elements = IC;

    // 一个 uint32_t 保存 8 个 INT4 权重
    const size_t weight_elements =
        static_cast<size_t>(OC) * IC / PACK_FACTOR;

    // 每个输出通道有 IC / 128 个量化组
    const int num_groups = IC / GROUP_SIZE;

    /*
     * TinyChatEngine gemv_kernel_g128 中：
     *
     * zeros_w =
     *     ceil(num_groups / 8)
     *
     * 一个 uint32_t 保存 8 个 4-bit zero point
     */
    const int zeros_w =
        make_divisible_host(num_groups, PACK_FACTOR);

    /*
     * scale 的第二维会补齐到 8 的整数倍
     *
     * sf_w = zeros_w * 8
     */
    const int sf_w = zeros_w * PACK_FACTOR;

    const size_t zero_elements =
        static_cast<size_t>(OC) * zeros_w;

    const size_t scale_elements =
        static_cast<size_t>(OC) * sf_w;

    const size_t output_elements = OC;

    // 字节数
    const size_t input_bytes =
        input_elements * sizeof(half);

    const size_t weight_bytes =
        weight_elements * sizeof(uint32_t);

    const size_t zero_bytes =
        zero_elements * sizeof(uint32_t);

    const size_t scale_bytes =
        scale_elements * sizeof(half);

    const size_t output_bytes =
        output_elements * sizeof(half);


    // ========================================================
    // 2. 分配 GPU Device Memory（设备内存）
    // ========================================================

    half* d_input = nullptr;
    uint32_t* d_weight = nullptr;
    uint32_t* d_zero = nullptr;
    half* d_scale = nullptr;
    half* d_output = nullptr;

    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&d_input),
        input_bytes));

    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&d_weight),
        weight_bytes));

    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&d_zero),
        zero_bytes));

    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&d_scale),
        scale_bytes));

    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&d_output),
        output_bytes));


    // ========================================================
    // 3. 初始化输入
    //
    // 为了方便验证：
    //
    // activation = 1
    //
    // INT4 weight = 9
    // zero point  = 8
    // scale       = 1
    //
    // 所以：
    //
    // dequantized_weight
    // = (9 - 8) * 1
    // = 1
    //
    // 最终：
    //
    // output = IC
    // ========================================================

    std::vector<half> h_input(input_elements);
    std::vector<half> h_scale(scale_elements);

    for (size_t i = 0; i < input_elements; ++i) {
        h_input[i] = __float2half(1.0f);
    }

    for (size_t i = 0; i < scale_elements; ++i) {
        h_scale[i] = __float2half(1.0f);
    }

    CUDA_CHECK(cudaMemcpy(
        d_input,
        h_input.data(),
        input_bytes,
        cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(
        d_scale,
        h_scale.data(),
        scale_bytes,
        cudaMemcpyHostToDevice));


    /*
     * 一个字节填 0x99：
     *
     * uint32_t:
     *
     * 0x99999999
     *
     * 每个 nibble（4 bit）都是 9
     *
     * 因此 8 个 INT4 都是 9。
     */
    CUDA_CHECK(cudaMemset(
        d_weight,
        0x99,
        weight_bytes));


    /*
     * zero point:
     *
     * 0x88888888
     *
     * 每个 4-bit zero point 都是 8。
     */
    CUDA_CHECK(cudaMemset(
        d_zero,
        0x88,
        zero_bytes));


    // 先把输出清零
    CUDA_CHECK(cudaMemset(
        d_output,
        0,
        output_bytes));


    // ========================================================
    // 4. 构造 TinyChatEngine 的 matmul_params
    // ========================================================

    matmul_params params{};

    // A = activation
    params.A.row = 1;
    params.A.column = IC;
    params.A.half_data_ptr = d_input;

    // B = packed INT4 weight
    params.B.row = IC / PACK_FACTOR;
    params.B.column = OC;
    params.B.int32_data_ptr =
        reinterpret_cast<int32_t*>(d_weight);

    // C = output
    params.C.row = 1;
    params.C.column = OC;
    params.C.half_data_ptr = d_output;

    // Scale
    params.half_scales = d_scale;

    // Zero point
    params.int32_zero_point =
        reinterpret_cast<int*>(d_zero);

    // W4A16 CUDA 使用 group size = 128
    params.block_size = GROUP_SIZE;


    // ========================================================
    // 5. 创建 TinyChatEngine GEMV Operator
    // ========================================================

    matmul::MatmulOperator op;


    // ========================================================
    // 6. Warmup（预热）
    //
    // 排除：
    // - CUDA Context 初始化
    // - Cache 冷启动
    // - 首次 Kernel Launch 等影响
    // ========================================================

    for (int i = 0; i < warmup_iters; ++i) {
        op.gemv_forward_cuda(&params);
    }

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());


    // ========================================================
    // 7. 正确性简单验证
    // ========================================================

    std::vector<half> h_output(output_elements);

    CUDA_CHECK(cudaMemcpy(
        h_output.data(),
        d_output,
        output_bytes,
        cudaMemcpyDeviceToHost));

    bool correct = true;

    const float expected = static_cast<float>(IC);

    // 只检查前 16 个输出即可
    const int check_num = std::min(16, OC);

    for (int i = 0; i < check_num; ++i) {
        const float value = __half2float(h_output[i]);

        if (std::fabs(value - expected) > 8.0f) {
            correct = false;

            std::cerr
                << "Correctness check failed at output["
                << i
                << "]: "
                << value
                << ", expected about "
                << expected
                << "\n";

            break;
        }
    }


    // ========================================================
    // 8. CUDA Event（事件）计时
    // ========================================================

    cudaEvent_t start;
    cudaEvent_t stop;

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));

    for (int i = 0; i < benchmark_iters; ++i) {
        op.gemv_forward_cuda(&params);
    }

    CUDA_CHECK(cudaEventRecord(stop));

    CUDA_CHECK(cudaEventSynchronize(stop));

    CUDA_CHECK(cudaGetLastError());


    // ========================================================
    // 9. 计算平均 Kernel 延迟
    // ========================================================

    float total_ms = 0.0f;

    CUDA_CHECK(cudaEventElapsedTime(
        &total_ms,
        start,
        stop));

    const float avg_ms =
        total_ms / benchmark_iters;

    const float avg_us =
        avg_ms * 1000.0f;


    // ========================================================
    // 10. 估算有效显存带宽
    //
    // 每次 GEMV 至少需要读取：
    //
    // weight
    // input
    // scale
    // zero
    //
    // 并写 output
    //
    // 这里只作为后续不同版本之间的统一比较指标。
    // ========================================================

    const double transferred_bytes =
        static_cast<double>(weight_bytes) +
        static_cast<double>(input_bytes) +
        static_cast<double>(scale_bytes) +
        static_cast<double>(zero_bytes) +
        static_cast<double>(output_bytes);

    const double seconds =
        static_cast<double>(avg_ms) / 1000.0;

    const double effective_bandwidth_GBs =
        transferred_bytes /
        seconds /
        1.0e9;


    // ========================================================
    // 11. 输出结果
    // ========================================================

    std::cout
        << "\n============================================\n";

    std::cout
        << "W4A16 GEMV Benchmark\n";

    std::cout
        << "IC                : "
        << IC
        << "\n";

    std::cout
        << "OC                : "
        << OC
        << "\n";

    std::cout
        << "Group size        : "
        << GROUP_SIZE
        << "\n";

    std::cout
        << "Warmup iterations : "
        << warmup_iters
        << "\n";

    std::cout
        << "Benchmark iters   : "
        << benchmark_iters
        << "\n";

    std::cout
        << "Weight size       : "
        << static_cast<double>(weight_bytes) / 1024.0 / 1024.0
        << " MiB\n";

    std::cout
        << "Average latency   : "
        << avg_us
        << " us\n";

    std::cout
        << "Average latency   : "
        << avg_ms
        << " ms\n";

    std::cout
        << "Effective BW      : "
        << effective_bandwidth_GBs
        << " GB/s\n";

    std::cout
        << "Correctness       : "
        << (correct ? "PASS" : "FAIL")
        << "\n";

    std::cout
        << "Output[0]         : "
        << __half2float(h_output[0])
        << "\n";

    std::cout
        << "Expected          : "
        << expected
        << "\n";

    std::cout
        << "============================================\n";


    // ========================================================
    // 12. 清理资源
    // ========================================================

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_weight));
    CUDA_CHECK(cudaFree(d_zero));
    CUDA_CHECK(cudaFree(d_scale));
    CUDA_CHECK(cudaFree(d_output));
}


// ============================================================
// main
// ============================================================
int main() {
    int device = 0;

    CUDA_CHECK(cudaSetDevice(device));

    cudaDeviceProp prop{};

    CUDA_CHECK(cudaGetDeviceProperties(
        &prop,
        device));

    std::cout
        << "GPU: "
        << prop.name
        << "\n";

    std::cout
        << "Compute Capability: "
        << prop.major
        << "."
        << prop.minor
        << "\n";


    // ========================================================
    // LLaMA2-7B 三个典型 Linear 形状
    // ========================================================

    // Attention O Projection 等
    benchmark_gemv(
        4096,
        4096
    );

    // MLP Gate / Up Projection
    benchmark_gemv(
        4096,
        11008
    );

    // MLP Down Projection
    benchmark_gemv(
        11008,
        4096
    );


    CUDA_CHECK(cudaDeviceReset());

    return 0;
}