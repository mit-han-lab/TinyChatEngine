#include "operators.h"
#include "utils.h"
#include "utils.h"

void load_Embedding_params_metal(Embedding_cuda& op, std::string prefix) {
    op.lookup.load((prefix + "/weight.bin").c_str());
}

// TODO: implement metal side
void Embedding_cuda::forward(Matrix3D<int> input_id, Matrix3D<half> output) {
    PROFILE_START(profile_name);
    assert(input_id.m_dim_x == 1);
    assert(input_id.m_dim_y == 1);
    assert(input_id.m_dim_z == output.m_dim_y);
    assert(output.m_dim_z == this->embed_dim);

    int threadsPerBlock = 1024;
    int blocksPerGrid = (input_id.m_dim_z + threadsPerBlock - 1) / threadsPerBlock;
    EmbeddingKernel_metal(input_id, output, this->lookup.m_data, this->embed_dim);

    PROFILE_END(profile_name);
}