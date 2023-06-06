#include "common.h"

template <typename T>
void add(Matrix3D<T> a, Matrix3D<T> b, Matrix3D<T> c);

void SiLuMul(Matrix3D<float> a, Matrix3D<float> b);

struct transpose_1_2idx_float_arg {
    int start_idx, end_idx;
    Matrix3D<float> input, output;
};

void transpose_1_2idx_float_threads(Matrix3D<float> &input, Matrix3D<float> &output, int num_thread);
