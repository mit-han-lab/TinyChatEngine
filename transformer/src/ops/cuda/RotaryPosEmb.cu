#include <cmath>
#include "operators.h"

// __global__ void RotaryPosEmb_float_forward(Matrix3D<float> query, Matrix3D<float> key, Matrix3D<float> cos, Matrix3D<float> sin, int start_idx, int len) {
//   // TODO: maybe we can use shared memory here
//   float query_buf[4096], key_buf[4096];

//   int num_heads = query.m_dim_x;
//   int head_embed = cos.m_dim_z;
//   int half = head_embed / 2;
  
//   // Convert the 1D CUDA thread indices into 3D indices
//   int b = blockIdx.x;
//   int i = threadIdx.x;

//   if(b < num_heads && i < len) {
//     for(int j = 0; j < half; j++) {
//       query_buf[j] = -1 * query(b, i, j + half);
//       key_buf[j] = -1 * key(b, i, j + half);
//     }
//     for(int j = half; j < head_embed; j++) {
//       query_buf[j] = query(b, i, j - half);
//       key_buf[j] = key(b, i, j - half);
//     }

//     for(int j = 0; j < head_embed; j++) {
//       query(b, i, j) = ((query(b, i, j) * cos(0, i + start_idx, j)) +
//                         (query_buf[j] * sin(0, i + start_idx, j)));
//       key(b, i, j) = ((key(b, i, j) * cos(0, i + start_idx, j)) +
//                       (key_buf[j] * sin(0, i + start_idx, j)));
//     }
//   }
// }

__global__ void RotaryPosEmb_cuda_forward(Matrix3D<half> query, Matrix3D<half> key, Matrix3D<half> cos, Matrix3D<half> sin, int start_idx, int len) {
  // TODO: maybe we can use shared memory here
  half query_buf[4096], key_buf[4096];

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

      query(b, i, j) = __hadd(__hmul(query(b, i, j), cos_half), __hmul(query_buf[j], sin_half));  // TODO: check if we can optimize this by using __hfma
      key(b, i, j) = __hadd(__hmul(key(b, i, j), cos_half), __hmul(key_buf[j], sin_half));  // TODO: check if we can optimize this by using __hfma
    }
  }
}
