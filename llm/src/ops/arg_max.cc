#include "ops/arg_max.h"

#include <cassert>

void arg_max_dim2(Matrix3D<float> &input, Matrix3D<int> &output) {
    int bz = input.m_dim_x;
    int sqlen = input.m_dim_y;
    int voc_size = input.m_dim_z;

    assert(sqlen == output.m_dim_z);
    assert(bz == output.m_dim_x);

    for (int b = 0; b < bz; b++) {
        for (int i = 0; i < sqlen; i++) {
            float max = FLOAT_MIN;
            int max_idx = -1;
            for (int j = 0; j < voc_size; j++) {
                float v = input(b, i, j);
                if (max < v) {
                    max = v;
                    max_idx = j;
                }
            }
            output(b, 0, i) = max_idx;
        }
    }
}
