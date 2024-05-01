#include <cmath>
#include <iomanip>
#include "operators.h"
#include "utils.h"
#include "metal_compute.h"

// TODO: modify metal for weights
void LlamaRMSNorm_metal::forward(const Matrix3D<half> &x, Matrix3D<half> &output, float eps) {
    const struct metal_params params;
    params.A.row = x.m_dim_y;
    params.A.column = x.m_dim_z;
    params.A.half_data_ptr = x.m_data;
    params.B.row = b.m_dim_z;     // k
    params.B.column = b.m_dim_y;  // n
    params.B.half_data_ptr = b.m_data;
    params.C.row = output.m_dim_y;
    params.C.column = output.m_dim_z;
    params.C.half_data_ptr = output.m_data;

    params.op = METAL_KERNEL_RMS_NORM;
    params.eps = eps;
    add_node(&params);
    return;
}