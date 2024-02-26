#include <cmath>
#include "operators.h"


// TODO: scale?
void softmax(Matrix3D<half> input, Matrix3D<half> output) {
    struct matmul_params params;
    params.A.row = input.m_dim_y;
    params.A.column = input.m_dim_z;
    params.A.half_data_ptr = input.m_data;
    params.C.row = output.m_dim_y;
    params.C.column = output.m_dim_z;
    params.C.half_data_ptr = output.m_data;

    matmul::MatmulOperator op = matmul::MatmulOperator();
    op.soft_max_metal(&params, input.m_dim_x, input.m_dim_y, input.m_dim_z, 1.0); 
}