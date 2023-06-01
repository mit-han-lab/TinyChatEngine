#include <cstring>

#include "operators.h"
#include "utils.h"

void load_Embedding_params(Embedding& op, std::string prefix) {
    op.lookup.load((prefix + "/weight.bin").c_str());
    // read_to_array((prefix + "/weight.bin").c_str(), op.lookup.m_data, op.lookup.length());
}

void Embedding::forward(Matrix3D<int> input_id, Matrix3D<float> output) {
    PROFILE_START(profile_name);
    assert(input_id.m_dim_x == 1);
    assert(input_id.m_dim_y == 1);
    assert(input_id.m_dim_z == output.m_dim_y);
    assert(output.m_dim_z == this->embed_dim);

    for (int i = 0; i < input_id.m_dim_z; i++) {
        int token_id = input_id(0, 0, i);
        float* output_sample_ptr = &output.m_data[i * this->embed_dim];
        float* target_embed = &this->lookup.m_data[token_id * this->embed_dim];
        memcpy(output_sample_ptr, target_embed, sizeof(float) * this->embed_dim);
    }
    PROFILE_END(profile_name);
}
