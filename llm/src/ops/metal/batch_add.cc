#include "operators.h"

// done
void batch_Add(const Matrix3D<float> &input, const Matrix3D<float> &input2, Matrix3D<float> &output) {
    struct matmul_params params;
    params.A.data_ptr = input.m_data;
    params.B.data_ptr = input2.m_data;
    params.C.data_ptr = output.m_data;

    matmul::MatmulOperator op = matmul::MatmulOperator();
    op.batch_add_metal(&params, input.m_dim_x, input.m_dim_y, input.m_dim_z);
}
