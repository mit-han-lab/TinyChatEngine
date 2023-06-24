#include <assert.h>
#include <sys/time.h>
#include <stdlib.h>
#include <cstdlib>
#include <iostream>

#include "../matmul.h"

#include <torch/torch.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <cuda_fp16.h>
#include <c10/cuda/CUDAGuard.h>
#include "gemm_cuda.h"
#include "dequantize.cuh"

const int threadDim = 32;
const int TILE_SIZE = threadDim;

static bool first_run = true;

__global__ void matrixMul_blockC(float *A, float *B, float *C, int A_row, int A_column, int B_column){
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;

	float acc = 0;
	for (int k = 0; k < A_column; k++)
		acc += A[j * A_column + k] * B[k * B_column + i];
	C[j * B_column +i] = acc;
}

__global__ void matrixMultiplyShared(const float *A, const float *B, float *C, int A_row, int A_column, int B_column) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	__shared__ float As[TILE_SIZE][TILE_SIZE];
	__shared__ float Bs[TILE_SIZE][TILE_SIZE];

	float value = 0;

	for (int i = 0; i < A_column / TILE_SIZE; i++){
		As[threadIdx.y][threadIdx.x] = A[(blockIdx.y * TILE_SIZE + threadIdx.y) * A_column + TILE_SIZE * i + threadIdx.x];
		Bs[threadIdx.y][threadIdx.x] = B[(i * TILE_SIZE + threadIdx.y) * B_column + blockIdx.x * TILE_SIZE + threadIdx.x];

		__syncthreads();

		for (int k = 0; k < TILE_SIZE; k++)
			value += As[threadIdx.y][k] * Bs[k][threadIdx.x];

		__syncthreads();
	}


	C[row * B_column + col] = value;
}


/* AWQ Implementation */

// Pack two half values.
static inline __device__ __host__ unsigned
__pack_half2(const half x, const half y) {
  unsigned v0 = *((unsigned short *)&x);
  unsigned v1 = *((unsigned short *)&y);
  return (v1 << 16) | v0;
}

__global__ void __launch_bounds__(64) gemm_forward_4bit_cuda_m16n128k32(int split_k_iters, half* __restrict__ A, int* __restrict__ B, half* __restrict__ scaling_factors, int* __restrict__ zeros, int M, int IC, int OC, half* __restrict__ C) 
{
  static constexpr uint32_t ZERO = 0x0;
  float C_warp[32];
  __shared__ half A_shared[16 * (32 + 8)];
  __shared__ half B_shared[32 * (128 + 8)];
  
  __shared__ half scaling_factors_shared[128];
  __shared__ half zeros_shared[128];

  int j_factors1 = ((OC + 128 - 1) / 128);

  int blockIdx_x = 0;
  int blockIdx_y = blockIdx.x % ((M + 16 - 1) / 16 * j_factors1);
  int blockIdx_z = blockIdx.x / ((M + 16 - 1) / 16 * j_factors1);

  half A_shared_warp[8];
  half B_shared_warp[32];
  for (int j_0_4_init = 0; j_0_4_init < 4; ++j_0_4_init) {
    for (int i = 0; i < 8; ++i) {
      C_warp[(j_0_4_init * 8) + i] = 0.0;
    }
  }

  static constexpr int row_stride_warp = 32 * 8 / 32;
  static constexpr int row_stride = 2 * 32 * 8 / 128;
  bool ld_zero_flag = (threadIdx.y * 32 + threadIdx.x) * 8 < 128;
  // TODO: Haotian: blockIdx_y / j_factors1 in A loading to support bsz > 16
  bool ld_A_flag = (blockIdx_y / j_factors1 * 16 + threadIdx.y * row_stride_warp + threadIdx.x * 8 / 32) < M;     // threadIdx.y is warp_id
  // bool wb_C_flag = (threadIdx.x / 4) < M;

  half* A_ptr = A 
                + (((int)blockIdx_y) / j_factors1 * 16 + (((int)threadIdx.y) * row_stride_warp) + ((int)threadIdx.x) / (32 / 8)) * IC
                + (((int)threadIdx.x) % (32 / 8)) * 8;
  
  int* B_ptr = B
            + ((int)threadIdx.y) * (OC / 8) * 2
            + (((int)threadIdx.x) / (128 / 8)) * (OC / 8)
            + (((int)blockIdx_y) % j_factors1) * (128 / 8)
            + (((int)threadIdx.x) % (128 / 8)) * 1;
                        
  half* A_shared_ptr = A_shared 
                    + ((int)threadIdx.y) * row_stride_warp * (32 + 8) 
                    + (((int)threadIdx.x) / (32 / 8)) * (32 + 8)
                    + (((int)threadIdx.x) % (32 / 8) ) * 8;

  half* B_shared_ptr = B_shared
                    + ((int)threadIdx.y) * (row_stride / 2) * (128 + 8)
                    + (((int)threadIdx.x) / (128 / 8)) * (128 + 8)
                    + (((int)threadIdx.x) % (128 / 8)) * 8;
  
  int* zeros_ptr = zeros
                + (((int)blockIdx_y) % j_factors1) * (128 / 8)
                + ((int)threadIdx.x) % (128 / 8);
  
  half* scaling_factors_ptr = scaling_factors
                            + (((int)blockIdx_y) % j_factors1) * (128) 
                            + (((int)threadIdx.x) % (128 / 8)) * 8;

  half* C_ptr = C 
              + blockIdx_z * M * OC        // blockIdz.x -> split_k dim
              + (((int)blockIdx_y) % j_factors1) * 128
              + ((int)threadIdx.y) * 64
              + (((int)threadIdx.x) % 4) * 2;

  // preload s.f. and zeros
  int k_bound = (IC / 32 + split_k_iters - 1) / split_k_iters;
  if ((k_bound - 1) * 32 + blockIdx_z >= IC) k_bound -= 1;
  for (int _k_0_0 = 0; _k_0_0 < k_bound; ++_k_0_0) {
    int k_0_0 = _k_0_0 * split_k_iters + blockIdx_z;
    __syncthreads();
    // TODO: Haotian: blockIdx_y / j_factors1 in A loading to support bsz > 16
    if (ld_A_flag)
    {
      *(uint4*)(A_shared_ptr) = *(uint4*)(A_ptr + (k_0_0 * 32));
    }
    else
    {
      *(uint4*)(A_shared_ptr) = make_uint4(0, 0, 0, 0);
    }

    // for (int ax0_ax1_fused_0 = 0; ax0_ax1_fused_0 < 2; ++ax0_ax1_fused_0) {
    uint32_t zeros_loaded = *(uint32_t*)(zeros_ptr + k_0_0 * 32 / 128 * (OC / 8));
    uint4 B_loaded_zero = dequantize_s4_to_fp16x2(zeros_loaded);
    uint4 B_loaded_scale = *(uint4*)(scaling_factors_ptr + k_0_0 * 32 / 128 * (OC));
    /*
    if (blockIdx_z == 0 && blockIdx_y == 0 && k_0_0 == 0 && threadIdx.x == 0 && threadIdx.y == 0){
      printf("%x %x %x %x %x %x %x %x\n", B_loaded_scale.x, B_loaded_scale.y, B_loaded_scale.z, B_loaded_scale.w, B_loaded_zero.x, B_loaded_zero.y, B_loaded_zero.z, B_loaded_zero.w);
    }
    */
    // uint4 B_loaded_scale = make_uint4(0, 0, 0, 0);
    int* B_ptr_local = B_ptr + k_0_0 * 32 * (OC / 8);

    for (int ax0_ax1_fused_0 = 0; ax0_ax1_fused_0 < 8; ++ax0_ax1_fused_0) {

      // B: 32 x 136 (128+8) float16
      // each warp: 32 x 4
      // each thr: read 32 bit -> convert to 8xFP16 (a UINT4) -> scale and minus zero -> WB UINT4
      // *(uint4*)(B_shared + ((((ax0_ax1_fused_0 * 544) + (((int)threadIdx.y) * 272)) + ((((int)threadIdx.x) >> 4) * 136)) + ((((int)threadIdx.x) & 15) * 8))) = *(uint4*)(B + ((((((k_0_0 * 163840) + (ax0_ax1_fused_0 * 20480)) + (((int)threadIdx.y) * 10240)) + ((((int)threadIdx.x) >> 4) * 5120)) + (((int)blockIdx_y) * 128)) + ((((int)threadIdx.x) & 15) * 8)));
      // row stride in shared memory: (NWARPS * 32 * 8 / cta_N) 
      uint32_t B_loaded = *(uint32_t*)(B_ptr_local + ax0_ax1_fused_0 * row_stride * (OC / 8));
      uint4 B_loaded_fp16 = dequantize_s4_to_fp16x2(B_loaded);
      //uint4 B_loaded_zero = *(uint4*)(zeros_shared + (threadIdx.x % (cta_N / 8)) * 8);

      // uint4 B_loaded_scale = *(uint4*)(scaling_factors_shared + (threadIdx.x % (cta_N / 8)) * 8);
      // - zero and * scale
      // TODO (Haotian): can save 4 assembly instructions if sormulate as deq = q * scale - zero * scale.
      asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.x) : "r"(B_loaded_fp16.x), "r"(B_loaded_zero.x));
      asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n" : "=r"(B_loaded_fp16.x) : "r"(B_loaded_fp16.x), "r"(B_loaded_scale.x), "r"(ZERO));
      asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.y) : "r"(B_loaded_fp16.y), "r"(B_loaded_zero.y));
      asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n" : "=r"(B_loaded_fp16.y) : "r"(B_loaded_fp16.y), "r"(B_loaded_scale.y), "r"(ZERO));
      asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.z) : "r"(B_loaded_fp16.z), "r"(B_loaded_zero.z));
      asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n" : "=r"(B_loaded_fp16.z) : "r"(B_loaded_fp16.z), "r"(B_loaded_scale.z), "r"(ZERO));
      asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.w) : "r"(B_loaded_fp16.w), "r"(B_loaded_zero.w));
      asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n" : "=r"(B_loaded_fp16.w) : "r"(B_loaded_fp16.w), "r"(B_loaded_scale.w), "r"(ZERO));
      /*
      if (ax0_ax1_fused_0 == 0 && blockIdx_z == 0 && blockIdx_y == 0 && k_0_0 == 0 && threadIdx.x == 17 && threadIdx.y == 0){
        printf("[x] %X %X %X %X\n", B_loaded_fp16.x, B_loaded_fp16.y, B_loaded_fp16.z, B_loaded_fp16.w);
      }
      */

      // write back
      *(uint4*)(B_shared_ptr + ax0_ax1_fused_0 * row_stride * (128 + 8)) = B_loaded_fp16;
    }
    __syncthreads();

    for (int k_0_1 = 0; k_0_1 < 2; ++k_0_1) {
      {
        unsigned int addr;
        __asm__ __volatile__(
          "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, addr; }\n"
          : "=r"(addr)
          : "l"((void *)((&(A_shared[(k_0_1 * 16)])) + (((((int)threadIdx.x) & 15) * 40) + ((((int)threadIdx.x) >> 4) * 8))))
        );


        __asm__ __volatile__(
          "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
          "{%0, %1, %2, %3}, [%4];\n"
          : "=r"(((unsigned *)(A_shared_warp + 0))[0]), "=r"(((unsigned *)(A_shared_warp + 0))[1]), "=r"(((unsigned *)(A_shared_warp + 0))[2]), "=r"(((unsigned *)(A_shared_warp + 0))[3])
          : "r"(addr)
        );
      }

      for (int ax1_0 = 0; ax1_0 < 4; ++ax1_0) {
        {
          unsigned int addr;
          __asm__ __volatile__(
            "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, addr; }\n"
            : "=r"(addr)
            : "l"((void *)((&(B_shared[(((k_0_1 * 2176) + (((int)threadIdx.y) * 64)) + (ax1_0 * 16))])) + (((((int)threadIdx.x) & 15) * 136) + ((((int)threadIdx.x) >> 4) * 8))))
          );
          __asm__ __volatile__(
            "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16"
            "{%0, %1, %2, %3}, [%4];\n"
            : "=r"(((unsigned *)(B_shared_warp + (ax1_0 * 8)))[0]), "=r"(((unsigned *)(B_shared_warp + (ax1_0 * 8)))[1]), "=r"(((unsigned *)(B_shared_warp + (ax1_0 * 8)))[2]), "=r"(((unsigned *)(B_shared_warp + (ax1_0 * 8)))[3])
            : "r"(addr)
          );
        }
      }
      for (int j_0_4 = 0; j_0_4 < 4; ++j_0_4) {
        {
          __asm__ __volatile__(
            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
            "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
            :  "=f"(((float *)(C_warp + (j_0_4 * 8)))[0]), "=f"(((float *)(C_warp + (j_0_4 * 8)))[1]), "=f"(((float *)(C_warp + (j_0_4 * 8)))[2]), "=f"(((float *)(C_warp + (j_0_4 * 8)))[3])
            : "r"(((unsigned *)(A_shared_warp + 0))[0]), "r"(((unsigned *)(A_shared_warp + 0))[1]), "r"(((unsigned *)(A_shared_warp + 0))[2]), "r"(((unsigned *)(A_shared_warp + 0))[3]), "r"(((unsigned *)(B_shared_warp + (j_0_4 * 8)))[0]), "r"(((unsigned *)(B_shared_warp + (j_0_4 * 8)))[1]), "f"(((float *)(C_warp + (j_0_4 * 8)))[0]), "f"(((float *)(C_warp + (j_0_4 * 8)))[1]), "f"(((float *)(C_warp + (j_0_4 * 8)))[2]), "f"(((float *)(C_warp + (j_0_4 * 8)))[3]));
        }

        {
          __asm__ __volatile__(
            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
            "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
            :  "=f"(((float *)(C_warp + ((j_0_4 * 8) + 4)))[0]), "=f"(((float *)(C_warp + ((j_0_4 * 8) + 4)))[1]), "=f"(((float *)(C_warp + ((j_0_4 * 8) + 4)))[2]), "=f"(((float *)(C_warp + ((j_0_4 * 8) + 4)))[3])
            : "r"(((unsigned *)(A_shared_warp + 0))[0]), "r"(((unsigned *)(A_shared_warp + 0))[1]), "r"(((unsigned *)(A_shared_warp + 0))[2]), "r"(((unsigned *)(A_shared_warp + 0))[3]), "r"(((unsigned *)(B_shared_warp + ((j_0_4 * 8) + 4)))[0]), "r"(((unsigned *)(B_shared_warp + ((j_0_4 * 8) + 4)))[1]), "f"(((float *)(C_warp + ((j_0_4 * 8) + 4)))[0]), "f"(((float *)(C_warp + ((j_0_4 * 8) + 4)))[1]), "f"(((float *)(C_warp + ((j_0_4 * 8) + 4)))[2]), "f"(((float *)(C_warp + ((j_0_4 * 8) + 4)))[3]));
        }
      }
    }
  }

// TODO: Shang: Hoist loop invariance.
  for (int ax1_0_1 = 0; ax1_0_1 < 4; ++ax1_0_1) {
    for (int local_id = 0; local_id < 8; ++local_id) {
      int row_offset = (((int)blockIdx_y) / j_factors1) * 16 + ((int)threadIdx.x) / 4 + (local_id % 4) / 2 * 8;
      if (row_offset < M)
      {
        *(C_ptr + ax1_0_1 * 16 + row_offset * OC + (local_id / 4) * 8 + local_id % 2) = __float2half(C_warp[(ax1_0_1 * 8) + local_id]);
      }
    }
  }
}

// in_feats: M, IC [float16]
// kernel: IC, OC // 8 [int32] -> cast to IC, OC [uint4b]
// scaling_factors: IC // G, OC [float16]
// zeros: IC // G, OC // 8 [int32] -> cast to IC // G, OC [uint4b]
// assume that batch_size < 16 for now

torch::Tensor gemm_forward_cuda(
    torch::Tensor _in_feats,
    torch::Tensor _kernel,
    torch::Tensor _scaling_factors,
    torch::Tensor _zeros,
    int split_k_iters)
{
    // std::cout << "0000" << std::endl;
    int num_in_feats = _in_feats.size(0);
    // std::cout << "num_in_feats = " << num_in_feats << std::endl;
    int num_in_channels = _in_feats.size(1);
    // std::cout << "num_in_channels = " << num_in_channels << std::endl;
    // std::cout << "a" << std::endl;
    // if (first_run) {
    //   _in_feats = _in_feats.to(torch::kCUDA);
    //   // _kernel = _kernel.to(torch::kCUDA);
    //   // _scaling_factors = _scaling_factors.to(torch::kCUDA);
    //   // _zeros = _zeros.to(torch::kCUDA);
    //   first_run = false;
    // }
    // _in_feats = _in_feats.to(torch::kCUDA);
    // std::cout << "?" << std::endl;
    
    // const at::cuda::OptionalCUDAGuard device_guard(device_of(_in_feats));
    
    // std::cout << "b" << std::endl;

    auto options = torch::TensorOptions().dtype(_in_feats.dtype()).device(_in_feats.device());
    // std::cout << "c" << std::endl;
    at::Tensor _out_feats = torch::empty({split_k_iters, num_in_feats, _kernel.size(1) * 8}, options);
    // std::cout << "d" << std::endl;
    int num_out_feats = _out_feats.size(-2);
    int num_out_channels = _out_feats.size(-1);

    // std::cout << "1" << std::endl;
    auto in_feats = reinterpret_cast<half*>(_in_feats.data_ptr<float>());
    // std::cout << "2" << std::endl;
    // // std::cout << _kernel << std::endl;
    auto kernel = reinterpret_cast<int*>(_kernel.data_ptr<int>());
    // std::cout << "3" << std::endl;
    auto out_feats = reinterpret_cast<half*>(_out_feats.data_ptr<float>());
    // std::cout << "4" << std::endl;
    auto scaling_factors = reinterpret_cast<half*>(_scaling_factors.data_ptr<float>());
    // std::cout << "5" << std::endl;
    auto zeros = reinterpret_cast<int*>(_zeros.data_ptr<int>());

    // std::cout << "BBBBB" << std::endl;

    if (num_out_channels % 128 != 0)
        throw std::invalid_argument("OC is not multiple of cta_N = 128");
    if (num_out_channels % 8 != 0)
        throw std::invalid_argument("OC is not multiple of pack_num = 8");
    int j_factors1 = num_out_channels / 128 / 1;
    dim3 num_blocks((num_out_feats + 16 - 1) / 16 * j_factors1 * split_k_iters);
    
    // threadIdx.x: 32
    // threadIdx.y: i_factors[2] * j_factors[2]
    dim3 threads_per_block(32, 2);
    // std::cout << "CCCCC" << std::endl;
    gemm_forward_4bit_cuda_m16n128k32<<<num_blocks, threads_per_block>>>(
        split_k_iters, in_feats, kernel, scaling_factors, zeros, num_in_feats, num_in_channels, num_out_channels, out_feats);

    // std::cout << "DDDDD" << std::endl;

    return _out_feats.sum(0);
}


namespace matmul{
  void MatmulOperator::mat_mul_accelerator_int4_fast(const struct matmul_params *params) {
		const struct matrix *A = &params->A, *B = &params->B, *C = &params->C;

    // std::cout << "mat_mul_accelerator_int4_fast -- A->row: " << A->row << " A->column: " << A->column 
              // << "; B->row: " << B->row << " B->column: " << B->column 
              // << "; C->row: " << C->row << " C->column: " << C->column << std::endl;
    
    // torch::Tensor out_feats = gemm_forward_cuda(
    //     torch::from_blob(A->data_ptr, {A->row, A->column}, torch::kHalf),
    //     torch::from_blob(B->data_ptr, {B->row, B->column}, torch::kInt),
    //     torch::from_blob(params->scales, {B->row / 128, B->column * 8}, torch::kHalf),
    //     torch::from_blob(params->int32_zero_point, {B->row / 128, B->column}, torch::kInt),
    //     8);
    
    auto option_fp = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, 1);
    auto option_int = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA, 1);
    torch::Tensor _in_feats = torch::from_blob(A->data_ptr, {A->row, A->column}, option_fp);
    torch::Tensor _kernel = torch::from_blob(B->data_ptr, {B->row, B->column}, option_int);
    torch::Tensor _scaling_factors = torch::from_blob(params->scales, {B->row / 128, B->column * 8}, option_fp);
    torch::Tensor _zeros = torch::from_blob(params->int32_zero_point, {B->row / 128, B->column}, option_int);

    torch::Tensor out_feats = gemm_forward_cuda(
        _in_feats,
        _kernel,
        _scaling_factors,
        _zeros,
        8);
    
    cudaMemcpy(C->data_ptr, out_feats.data_ptr(), C->column * C->row * sizeof(float), cudaMemcpyDeviceToHost);
  };

  void MatmulOperator::mat_mul_accelerator_int4_fast_no_offset(const struct matmul_params *params) {
		const struct matrix *A = &params->A, *B = &params->B, *C = &params->C;

    // std::cout << "mat_mul_accelerator_int4_fast_no_offset -- A->row: " << A->row << " A->column: " << A->column 
              // << "; B->row: " << B->row << " B->column: " << B->column 
              // << "; C->row: " << C->row << " C->column: " << C->column << std::endl;
    
    // std::cout << "AAAAA" << std::endl;
    
    // torch::Tensor out_feats = gemm_forward_cuda(
    //     torch::from_blob(A->data_ptr, {A->row, A->column}, torch::kHalf),
    //     torch::from_blob(B->data_ptr, {B->row, B->column}, torch::kInt),
    //     torch::from_blob(params->scales, {B->row / 128, B->column * 8}, torch::kHalf),
    //     torch::from_blob(params->int32_zero_point, {B->row / 128, B->column}, torch::kInt),
    //     8);
    
    auto option_fp = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, 1);
    auto option_int = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA, 1);
    // std::cout << "jcuycu" << std::endl;
    //torch::Tensor _in_feats = torch::from_blob(A->data_ptr, {A->row, A->column}, option_fp);
    torch::Tensor _in_feats = torch::from_blob(A->data_ptr, {A->row, A->column}).to(torch::kCUDA);
    // std::cout << "kchy" << std::endl;
    // torch::Tensor _kernel = torch::from_blob(B->int32_data_ptr, {B->row, B->column}, option_int);
    torch::Tensor _kernel = torch::from_blob(B->int32_data_ptr, {B->row, B->column}, torch::kInt).to(torch::kCUDA);
    // std::cout << "qued" << std::endl;
    // torch::Tensor _scaling_factors = torch::from_blob(params->scales, {B->row / 128, B->column * 8}, option_fp);
    torch::Tensor _scaling_factors = torch::from_blob(params->scales, {B->row / 128, B->column * 8}).to(torch::kCUDA);
    // std::cout << "-cud" << std::endl;
    // torch::Tensor _zeros = torch::from_blob(params->int32_zero_point, {B->row / 128, B->column}, option_int);
    torch::Tensor _zeros = torch::from_blob(params->int32_zero_point, {B->row / 128, B->column}, torch::kInt).to(torch::kCUDA);
    // std::cout << "msadi" << std::endl;

    torch::Tensor out_feats = gemm_forward_cuda(
        _in_feats,
        _kernel,
        _scaling_factors,
        _zeros,
        8);
    
    cudaMemcpy(C->data_ptr, out_feats.data_ptr(), C->column * C->row * sizeof(float), cudaMemcpyDeviceToHost);
  };

	void MatmulOperator::mat_mul_cuda(const struct matmul_params *params){
		const struct matrix *A = &params->A, *B = &params->B, *C = &params->C;
		assert(A->column == B->row);
		assert(C->column == B->column);
		assert(C->row == A->row);

		float *d_A;
		float *d_B;
		float *d_C;

		// Initailize C
		/*for (int i = 0; i < C->row; i++)
		  for (int j = 0; j < C->column; j++)
		  C->data_ptr[j + C->column * i] = 0;*/

		// Allocate memory
		cudaMalloc(&d_A, A->column*A->row*sizeof(float));
		cudaMalloc(&d_B, B->column*B->row*sizeof(float));
		cudaMalloc(&d_C, C->column*C->row*sizeof(float));

		// Copy data to GPU
		cudaMemcpy(d_A, A->data_ptr, A->column*A->row*sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(d_B, B->data_ptr, B->column*B->row*sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(d_C, C->data_ptr, C->column*C->row*sizeof(float), cudaMemcpyHostToDevice);

		// Make sure we can break the input matrix into blocks
		assert(A->column % threadDim == 0);
		assert(A->row % threadDim == 0);
		assert(B->column % threadDim == 0);
		const dim3 threadsPerBlock(threadDim, threadDim);
		const dim3 numBlocks(C->column / threadsPerBlock.x, C->row / threadsPerBlock.y);

		// Invoke the cuda imp.

		// struct timeval start, end;
		// gettimeofday(&start, NULL);
		//matrixMul_blockC<<< numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, A->row, A->column, B->column);
		matrixMultiplyShared<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, A->row, A->column, B->column);
		cudaDeviceSynchronize();
		// gettimeofday(&end, NULL);
		// int us = interval_to_us(&start, &end);
		// // std::cout << "cuda kernel: " << us / 1000 << " ms" << std::endl;

		// Get the result back
		cudaMemcpy(C->data_ptr, d_C, C->column*C->row*sizeof(float), cudaMemcpyDeviceToHost);
	}
}  // namespace matmul
