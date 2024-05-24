#include <cmath>
#include "../../../include/operators.h"
#include "utils.h"
#include "metal_compute.h"


void softmax_metal(Matrix3D<half> input, Matrix3D<half> output) {
    struct metal_params params;
    params.A.row = input.m_dim_y;
    params.A.column = input.m_dim_z;
    params.A.half_data_ptr = input.m_data;
    params.B.row = input.m_dim_z;     // k
    params.B.column = input.m_dim_y;  // n
    params.B.half_data_ptr = input.m_data;
    params.C.row = output.m_dim_y;
    params.C.column = output.m_dim_z;
    params.C.half_data_ptr = output.m_data;
    params.scale = 0.088388;
    params.op = METAL_KERNEL_SOFT_MAX;
    add_node(&params);
}