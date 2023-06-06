#include "llama_utils.h"

#include "common.h"
#include "utils.h"

template <typename T>
void add(Matrix3D<T> a, Matrix3D<T> b, Matrix3D<T> c) {
    PROFILE_START("int4llamaDecoderLayer::add");
    assert(c.length() == a.length() && a.length() == b.length());

    for (int i = 0; i < a.length(); i++) {
        c.m_data[i] = a.m_data[i] + b.m_data[i];
    }
    PROFILE_END("int4llamaDecoderLayer::add");
}

void SiLuMul(Matrix3D<float> a, Matrix3D<float> b) {
    PROFILE_START("MulSiLu");
    for (int i = 0; i < a.length(); i++) {
        float v = a.m_data[i];
        float silu_v = v * (1.0 / (1.0 + exp(-1 * v)));
        a.m_data[i] = silu_v * b.m_data[i];
    }
    PROFILE_END("MulSiLu");
}

void *transpose_1_2idx_float_func(void *args_) {
    struct transpose_1_2idx_float_arg *args = (struct transpose_1_2idx_float_arg *)args_;

    Matrix3D<float> input = args->input;
    Matrix3D<float> output = args->output;

    for (int i = 0; i < input.m_dim_x; i++) {
        for (int j = 0; j < input.m_dim_y; j++) {
            for (int k = args->start_idx; k < args->end_idx; k++) {
                output.m_data[i * output.m_dim_y * output.m_dim_z + k * output.m_dim_z + j] =
                    input.m_data[i * input.m_dim_y * input.m_dim_z + j * input.m_dim_z + k];
            }
        }
    }
    return NULL;
}

void transpose_1_2idx_float_threads(Matrix3D<float> &input, Matrix3D<float> &output, int num_thread) {
    PROFILE_START("transpose_1_2idx_float");
    assert(input.m_dim_x == output.m_dim_x);
    assert(input.m_dim_y == output.m_dim_z);
    assert(input.m_dim_z == output.m_dim_y);

    if (input.m_dim_y == 1 || input.m_dim_z == 1) {
        memcpy(output.m_data, input.m_data, input.length() * sizeof(float));
    } else {
        int num_thread = num_thread;
        int loop_over_dim = input.m_dim_z;
        if (num_thread > loop_over_dim) num_thread = loop_over_dim;

        pthread_t thread_pool[num_thread];
        struct transpose_1_2idx_float_arg threads_args[num_thread];

        // Thread creation
        for (int j = 0; j < num_thread; j++) {
            threads_args[j].start_idx = j * (loop_over_dim / num_thread);
            threads_args[j].input = input;
            threads_args[j].output = output;
            if (j == num_thread - 1)
                threads_args[j].end_idx = loop_over_dim;
            else
                threads_args[j].end_idx = (j + 1) * (loop_over_dim / num_thread);
            pthread_create(&thread_pool[j], NULL, transpose_1_2idx_float_func, &threads_args[j]);
        }
        // Join threads
        for (int j = 0; j < num_thread; j++) {
            pthread_join(thread_pool[j], NULL);
        }
    }

    PROFILE_END("transpose_1_2idx_float");
}

template void add<float>(Matrix3D<float> a, Matrix3D<float> b, Matrix3D<float> c);
