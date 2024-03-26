#include "operators.h"
#include "metal_compute.h"

// done
void batch_Add(const Matrix3D<float> &input, const Matrix3D<float> &input2, Matrix3D<float> &output) {
    const struct metal_params params;

    params.A.row = input.m_dim_y;
    params.A.column = input.m_dim_z;
    params.A.fp16_data_ptr = input.m_data;
    params.B.row = input2.m_dim_z;
    params.B.column = input2.m_dim_y;
    params.B.int32_data_ptr = input2.m_data;
    params.C.row = output.m_dim_y;
    params.C.column = output.m_dim_z;
    params.C.fp16_data_ptr = output.m_data;
    params.A.data_ptr = input.m_data;
    params.B.data_ptr = input2.m_data;
    params.C.data_ptr = output.m_data;
    struct metal_cgraph *graph = new (struct metal_cgraph);
    graph->n_nodes = 1;
    graph->mm_nodes[0] = params;
    metal_graph_compute(METAL_KERNEL_BATCH_ADD, graph);
}
