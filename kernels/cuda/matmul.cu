#include <assert.h>
#include <sys/time.h>
#include <stdlib.h>
#include <cstdlib>
#include <iostream>

#include "../matmul.h"

#include <cuda_runtime.h>
#include <torch/extension.h>
#include "gemm_cuda.h"
#include "dequantize.cuh"
#include <cuda_fp16.h>
#include <c10/cuda/CUDAGuard.h>

const int threadDim = 32;
const int TILE_SIZE = threadDim;

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

namespace matmul{

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
		matrixMultiplyShared<<< numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, A->row, A->column, B->column);
		cudaDeviceSynchronize();
		// gettimeofday(&end, NULL);
		// int us = interval_to_us(&start, &end);
		// std::cout << "cuda kernel: " << us / 1000 << " ms" << std::endl;

		// Get the result back
		cudaMemcpy(C->data_ptr, d_C, C->column*C->row*sizeof(float), cudaMemcpyDeviceToHost);
	}
}
