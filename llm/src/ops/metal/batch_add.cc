#include "operators.h"

void batch_Add(const Matrix3D<float> &input, const Matrix3D<float> &input2, Matrix3D<float> &output) {
    struct matmul_params params;
    params.A.row = input.m_dim_y;
    params.A.column = input.m_dim_z;
    params.A.data_ptr = input.m_data;
    params.B.row = input.m_dim_z;     // k
    params.B.column = input2.m_dim_y;  // n
    params.B.data_ptr = input2.m_data;
    params.C.row = output.m_dim_y;
    params.C.column = output.m_dim_z;
    params.C.data_ptr = output.m_data;

    matmul::MatmulOperator op = matmul::MatmulOperator();
    op.batch_add_metal(&params);
}
