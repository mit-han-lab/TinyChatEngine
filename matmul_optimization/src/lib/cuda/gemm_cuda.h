// #include <torch/extension.h>

// torch::Tensor gemm_forward_cuda_origin(torch::Tensor _in_feats, torch::Tensor _kernel,
//     torch::Tensor _scaling_factors, torch::Tensor _zeros, int split_k_iters);

void gemm_forward_cuda(const struct matmul_params *params, int split_k_iters);
