#include "../../../include/operators.h"
#include "utils.h"
#include "metal_compute.h"

void load_Embedding_params_metal(Embedding_metal& op, std::string prefix) {
    op.lookup.load((prefix + "/weight.bin").c_str());
}

// TODO: implement metal side
void Embedding_metal::forward(Matrix3D<int> input_id, Matrix3D<half> output) {
    PROFILE_START(profile_name);
    assert(input_id.m_dim_x == 1);
    assert(input_id.m_dim_y == 1);
    assert(input_id.m_dim_z == output.m_dim_y);
    assert(output.m_dim_z == this->embed_dim);

    const struct metal_params params;

    params.A.int32_data_ptr = input.m_data;
    params.B.data_ptr = this->lookup.m_data;
    params.C.half_data_ptr = output.m_data;
    params.op = METAL_KERNEL_EMBEDDING;
    params.embed_dim = this->embed_dim;
    add_node(&params);
    PROFILE_END(profile_name);
}