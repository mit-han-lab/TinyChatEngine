#include "operators.h"
#include "utils.h"
#include "metal_compute.h"

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

    struct metal_constants op_constants = new (struct metal_constants);
    op_constants.embed_dim = this->embed_dim;
    struct metal_cgraph *graph = new (struct metal_cgraph);
    graph->n_nodes = 1;
    graph->input_id = input_id;
    graph->output = output;
    graph->lookup = this->lookup.m_data;
    metal_graph_compute(METAL_KERNEL_EMBEDDING, graph);
    PROFILE_END(profile_name);
}