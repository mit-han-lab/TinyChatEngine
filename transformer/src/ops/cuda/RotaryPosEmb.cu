#include <cmath>
#include "operators.h"

__global__ void RotaryPosEmb_cuda_forward(Matrix3D<half> query, Matrix3D<half> key, Matrix3D<half> cos, Matrix3D<half> sin, int start_idx, int len) {
  half query_buf[128], key_buf[128];

  int num_heads = query.m_dim_x;
  int head_embed = cos.m_dim_z;
  int half_pos = head_embed / 2;
  
  // Convert the 1D CUDA thread indices into 3D indices
  int b = blockIdx.x;
  int i = threadIdx.x;

  if(b < num_heads && i < len) {
    for(int j = 0; j < half_pos; j++) {
      query_buf[j] = __hneg(query(b, i, j + half_pos));
      key_buf[j] = __hneg(key(b, i, j + half_pos));
    }

    for(int j = half_pos; j < head_embed; j++) {
      query_buf[j] = query(b, i, j - half_pos);
      key_buf[j] = key(b, i, j - half_pos);
    }

    for(int j = 0; j < head_embed; j++) {
      half cos_half = cos(0, i + start_idx, j);
      half sin_half = sin(0, i + start_idx, j);

      query(b, i, j) = __hfma(query(b, i, j), cos_half, __hmul(query_buf[j], sin_half));
      key(b, i, j) = __hfma(key(b, i, j), cos_half, __hmul(key_buf[j], sin_half));
    }
  }
}

__global__ void RotaryPosEmb_cuda_forward_shared(Matrix3D<half> query, Matrix3D<half> key, Matrix3D<half> cos, Matrix3D<half> sin, int start_idx, int len) {
  extern __shared__ half shared_memory[];

  half *query_buf = &shared_memory[0];
  half *key_buf = &shared_memory[4096];

  int num_heads = query.m_dim_x;
  int head_embed = cos.m_dim_z;
  int half_pos = head_embed / 2;

  int b = blockIdx.x;
  int i = threadIdx.x;

  if(b < num_heads && i < len) {
    // Load data into shared memory for faster access.
    for(int j = 0; j < half_pos; j++) {
      query_buf[threadIdx.x * head_embed + j] = __hneg(query(b, i, j + half_pos));
      key_buf[threadIdx.x * head_embed + j] = __hneg(key(b, i, j + half_pos));
    }

    for(int j = half_pos; j < head_embed; j++) {
      query_buf[threadIdx.x * head_embed + j] = query(b, i, j - half_pos);
      key_buf[threadIdx.x * head_embed + j] = key(b, i, j - half_pos);
    }

    __syncthreads();  // Synchronize to ensure all data is loaded before processing.

    for(int j = 0; j < head_embed; j++) {
      half cos_half = cos(0, i + start_idx, j);
      half sin_half = sin(0, i + start_idx, j);

      // Use the __hfma intrinsic function for faster multiply-add operations.
      query(b, i, j) = __hfma(query(b, i, j), cos_half, __hmul(query_buf[threadIdx.x * head_embed + j], sin_half));
      key(b, i, j) = __hfma(key(b, i, j), cos_half, __hmul(key_buf[threadIdx.x * head_embed + j], sin_half));
    }
  }
}
