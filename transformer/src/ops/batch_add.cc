#include "operators.h"

void batch_Add(const Matrix3D<float> &input, const Matrix3D<float> &input2,Matrix3D<float> &output) {
    PROFILE_START("batch_Add");
    assert(input.m_dim_y == input2.m_dim_y);
    assert(input.m_dim_z == input2.m_dim_z);
    assert(input.m_dim_x == output.m_dim_x);
    assert(input.m_dim_y == output.m_dim_y);
    assert(input.m_dim_z == output.m_dim_z);

    if (input.m_dim_x != input2.m_dim_x && input2.m_dim_x == 1) {
        // Find the maximum value in the input array
        for (int i = 0; i < input.m_dim_x; i++) {
            for (int j = 0; j < input.m_dim_y; j++) {
                for (int k = 0; k < input.m_dim_z; k++){
                    output(i, j, k) = input(i, j, k) + input2(0, j, k);
                }
            }
        }
    } else {
        throw("Unsupported dimension for softmax");
    }
    PROFILE_END("batch_Add");
}