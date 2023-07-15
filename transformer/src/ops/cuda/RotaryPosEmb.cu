#include "operators.cuh"
#include <cmath>

// TODO: optimize this with multithreading
__global__ void RotaryPosEmb_half_forward(Matrix3D_cuda<float> query, Matrix3D_cuda<float> key, Matrix3D_cuda<float> cos, Matrix3D_cuda<float> sin, int start_idx, int len) {
  __shared__ float query_buf[4096], key_buf[4096];

  int num_heads = query.m_dim_x;
  int head_embed = cos.m_dim_z;
  int max_sqlen = cos.m_dim_y;

  int half = head_embed / 2;
  int total_threads = blockDim.x * gridDim.x;
  int global_thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  
  // Convert the 1D CUDA thread indices into 3D indices
  int b = global_thread_id / (head_embed * len);
  int i = (global_thread_id / head_embed) % len;
  int j = global_thread_id % head_embed;

  if (b < num_heads && i < len && j < head_embed) {
    if (j < half) {
      // query_buf[j] = -1 * query[(b * len + i) * head_embed + j + half];
      // key_buf[j] = -1 * key[(b * len + i) * head_embed + j + half];
      query_buf[j] = -1 * query(b, i, j + half);
      key_buf[j] = -1 * key(b, i, j + half);
    }
    else {
      // query_buf[j] = query[(b * len + i) * head_embed + j - half];
      // key_buf[j] = key[(b * len + i) * head_embed + j - half];
      query_buf[j] = query(b, i, j - half);
      key_buf[j] = key(b, i, j - half);
    }

    query(b, i, j) = ((query(b, i, j) * cos(0, i + start_idx, j)) +
                      (query_buf[j] * sin(0, i + start_idx, j)));
    key(b, i, j) = ((key(b, i, j) * cos(0, i + start_idx, j)) +
                    (key_buf[j] * sin(0, i + start_idx, j)));
  }

  // for (int b = 0; b < num_heads; b++) {
  //   for (int i = 0; i < len; i++) {
  //     // first half
  //     for (int j = 0; j < half; j++) {
  //       query_buf[j] = -1 * query(b, i, j + half);
  //       key_buf[j] = -1 * key(b, i, j + half);
  //     }
  //     // second half
  //     for (int j = half; j < head_embed; j++) {
  //       query_buf[j] = query(b, i, j - half);
  //       key_buf[j] = key(b, i, j - half);
  //     }

  //     for (int j = 0; j < head_embed; j++) {
  //       query(b, i, j) = ((query(b, i, j) * cos(0, i + start_idx, j)) +
  //                         (query_buf[j] * sin(0, i + start_idx, j)));
  //       key(b, i, j) = ((key(b, i, j) * cos(0, i + start_idx, j)) +
  //                       (key_buf[j] * sin(0, i + start_idx, j)));
  //     }
  //   }
  // }
}
