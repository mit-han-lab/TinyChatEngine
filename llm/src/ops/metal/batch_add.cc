#include "operators.h"
#include "utils.h"
#include "metal_compute.h"

void batch_Add_metal(const Matrix3D<half> &input, const Matrix3D<half> &input2, Matrix3D<half> &output) {
    struct metal_params params;

    params.A.row = input.m_dim_y;
    params.A.column = input.m_dim_z;
    params.A.fp16_data_ptr = input.m_data;
    params.B.row = input2.m_dim_z;
    params.B.column = input2.m_dim_y;
    params.B.fp16_data_ptr = input2.m_data;
    params.C.row = output.m_dim_y;
    params.C.column = output.m_dim_z;
    params.C.fp16_data_ptr = output.m_data;
    params.op = METAL_KERNEL_BATCH_ADD;
    add_node(&params);
}
