#include <cstring>

#include "operators.cuh"
#include "utils.h"

__global__ void EmbeddingKernel(Matrix3D_cuda<int> input_id, Matrix3D_cuda<float> output, float* lookup, int embed_dim) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < input_id.m_dim_z) {
        int token_id = input_id(0, 0, i);
        float* output_sample_ptr = &output.m_data[i * embed_dim];
        float* target_embed = &lookup[token_id * embed_dim];

        for (int j = 0; j < embed_dim; ++j) {
            output_sample_ptr[j] = target_embed[j];
        }
    }
}


void load_Embedding_params(Embedding_half& op, std::string prefix) {
    op.lookup.load((prefix + "/weight.bin").c_str());
    // read_to_array((prefix + "/weight.bin").c_str(), op.lookup.m_data, op.lookup.length());
}

void Embedding_half::forward(Matrix3D_cuda<int> input_id, Matrix3D_cuda<float> output) {
    PROFILE_START(profile_name);
    assert(input_id.m_dim_x == 1);
    assert(input_id.m_dim_y == 1);
    assert(input_id.m_dim_z == output.m_dim_y);
    assert(output.m_dim_z == this->embed_dim);

    int threadsPerBlock = 256;
    int blocksPerGrid = (input_id.m_dim_z + threadsPerBlock - 1) / threadsPerBlock;
    EmbeddingKernel<<<blocksPerGrid, threadsPerBlock>>>(input_id, output, this->lookup.m_data, this->embed_dim);

    PROFILE_END(profile_name);
}
