#ifndef COMMON_CUDA_H
#define COMMON_CUDA_H
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdexcept>

#include "model.h"
#include "utils.h"

// #include <cuda.h>
// #include <cuda_fp16.h>
// #include <cuda_runtime.h>


template <typename T>
class Matrix3D_cuda {
   public:
    Matrix3D_cuda(T *data, int dim_x, int dim_y, int dim_z) : m_data(data), m_dim_x(dim_x), m_dim_y(dim_y), m_dim_z(dim_z) {}

    __host__ __device__ T &operator()(int x, int y, int z) {
        // if (x < 0 || x >= m_dim_x || y < 0 || y >= m_dim_y || z < 0 || z >= m_dim_z) {
        //     printf("%d, %d, %d\n", x, y, z);
        //     printf("%d, %d, %d\n", m_dim_x, m_dim_y, m_dim_z);
        //     throw std::out_of_range("Matrix3D: Indices out of range.");
        // }
        return m_data[x * m_dim_y * m_dim_z + y * m_dim_z + z];
    }

    __host__ __device__ const T &operator()(int x, int y, int z) const {
        // if (x < 0 || x >= m_dim_x || y < 0 || y >= m_dim_y || z < 0 || z >= m_dim_z) {
        //     printf("%d, %d, %d\n", x, y, z);
        //     printf("%d, %d, %d\n", m_dim_x, m_dim_y, m_dim_z);
        //     throw std::out_of_range("Matrix3D: Indices out of range.");
        // }
        return m_data[x * m_dim_y * m_dim_z + y * m_dim_z + z];
    }

    bool operator==(const Matrix3D_cuda<T> &other) const {
        if (m_dim_x != other.m_dim_x || m_dim_y != other.m_dim_y || m_dim_z != other.m_dim_z) {
            return false;
        }

        for (int x = 0; x < m_dim_x; ++x) {
            for (int y = 0; y < m_dim_y; ++y) {
                for (int z = 0; z < m_dim_z; ++z) {
                    if ((*this)(x, y, z) != other(x, y, z)) {
                        return false;
                    }
                }
            }
        }

        return true;
    }

    __host__ __device__ int length() const { return m_dim_x * m_dim_y * m_dim_z; }
    T sum() const {
        T sum = 0;
        for (int i = 0; i < this->length(); i++) {
            sum += this->m_data[i];
        }
        return sum;
    }
    T sum(int size) const {
        T sum = 0;
        for (int i = 0; i < size; i++) {
            sum += this->m_data[i];
        }
        return sum;
    }

    T sum(int size, int start_idx) const {
        T sum = 0;
        for (int i = 0; i < size; i++) {
            sum += this->m_data[start_idx + i];
        }
        return sum;
    }

    void load(const char *path) {
        std::ifstream infile(path, std::ios::binary | std::ios::in);
        if (infile.fail()) {
            std::cout << strerror(errno) << ": " << path << std::endl;
            throw("Expected error...");
        } else {
            infile.read(reinterpret_cast<char *>(this->m_data), this->length() * sizeof(T));
            infile.close();
        }
    }
    int m_dim_x, m_dim_y, m_dim_z;
    T *m_data;

    // Default constructor
    Matrix3D_cuda() { m_data = NULL; }
};

#endif