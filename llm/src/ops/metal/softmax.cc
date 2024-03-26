#include <cmath>
#include "operators.h"


void softmax(Matrix3D<half> input, Matrix3D<half> output) {
    const struct metal_params params;
    params.A.row = input.m_dim_y;
    params.A.column = input.m_dim_z;
    params.A.half_data_ptr = input.m_data;
    params.B.row = input.m_dim_z;     // k
    params.B.column = input.m_dim_y;  // n
    params.B.data_ptr = input.m_data;
    params.C.row = output.m_dim_y;
    params.C.column = output.m_dim_z;
    params.C.data_ptr = output.m_data;

    struct metal_constants op_constants = new (struct metal_constants);
    op_constants.scales = this->scales;
    struct metal_cgraph *graph = new (struct metal_cgraph);
    graph->n_nodes = 1;
    graph->mm_nodes[0] = params;
    metal_graph_compute(METAL_KERNEL_SOFT_MAX, graph);
}