#ifndef COMMON_H
#define COMMON_H
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdexcept>

#include "model.h"

#define MAX_LINEAR_LENGTH 1024 * 1024 * 16  // 16MB, TO BE REMOVED with better memory allocation!
#define DEBUG false

#define DEBUG_INS(x) \
    if (DEBUG) x

#ifdef QM_CUDA
#define QK 128
#else
#define QK 32
#endif

struct pack_q4_tensor {
    uint8_t qx[QK / 2];
    float scale;
};

struct pack_q8_tensor {
    int8_t qx[QK];
    float scale;
};

template <typename T>
class Matrix3D {
   public:
    Matrix3D(T *data, int dim_x, int dim_y, int dim_z) : m_data(data), m_dim_x(dim_x), m_dim_y(dim_y), m_dim_z(dim_z) {}

#if defined(__CUDACC__)
    __host__ __device__ T &operator()(int x, int y, int z) { return m_data[x * m_dim_y * m_dim_z + y * m_dim_z + z]; }

    __host__ __device__ const T &operator()(int x, int y, int z) const {
        return m_data[x * m_dim_y * m_dim_z + y * m_dim_z + z];
    }
#else
    T &operator()(int x, int y, int z) {
        if (x < 0 || x >= m_dim_x || y < 0 || y >= m_dim_y || z < 0 || z >= m_dim_z) {
            printf("%d, %d, %d\n", x, y, z);
            printf("%d, %d, %d\n", m_dim_x, m_dim_y, m_dim_z);
            throw std::out_of_range("Matrix3D: Indices out of range.");
        }
        return m_data[x * m_dim_y * m_dim_z + y * m_dim_z + z];
    }

    const T &operator()(int x, int y, int z) const {
        if (x < 0 || x >= m_dim_x || y < 0 || y >= m_dim_y || z < 0 || z >= m_dim_z) {
            printf("%d, %d, %d\n", x, y, z);
            printf("%d, %d, %d\n", m_dim_x, m_dim_y, m_dim_z);
            throw std::out_of_range("Matrix3D: Indices out of range.");
        }
        return m_data[x * m_dim_y * m_dim_z + y * m_dim_z + z];
    }
#endif

    bool operator==(const Matrix3D<T> &other) const {
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

#if defined(__CUDACC__)
    __host__ __device__ int length() const { return m_dim_x * m_dim_y * m_dim_z; }
#else
    int length() const { return m_dim_x * m_dim_y * m_dim_z; }
#endif

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
    T *m_data;
    int m_dim_x, m_dim_y, m_dim_z;

    // Default constructor
    Matrix3D() { m_data = NULL; }
};

template <typename T>
class Matrix4D {
   public:
    Matrix4D(T *data, int dim_w, int dim_x, int dim_y, int dim_z) : 
        m_data(data), m_dim_w(dim_w), m_dim_x(dim_x), m_dim_y(dim_y), m_dim_z(dim_z) {}

#if defined(__CUDACC__)
    __host__ __device__ T &operator()(int w, int x, int y, int z) { 
        return m_data[w * m_dim_x * m_dim_y * m_dim_z + x * m_dim_y * m_dim_z + y * m_dim_z + z]; 
    }

    __host__ __device__ const T &operator()(int w, int x, int y, int z) const {
        return m_data[w * m_dim_x * m_dim_y * m_dim_z + x * m_dim_y * m_dim_z + y * m_dim_z + z];
    }
#else
    T &operator()(int w, int x, int y, int z) {
        if (w < 0 || w >= m_dim_w || x < 0 || x >= m_dim_x || y < 0 || y >= m_dim_y || z < 0 || z >= m_dim_z) {
            printf("%d, %d, %d, %d\n", w, x, y, z);
            printf("%d, %d, %d, %d\n", m_dim_w, m_dim_x, m_dim_y, m_dim_z);
            throw std::out_of_range("Matrix4D: Indices out of range.");
        }
        return m_data[w * m_dim_x * m_dim_y * m_dim_z + x * m_dim_y * m_dim_z + y * m_dim_z + z];
    }

    const T &operator()(int w, int x, int y, int z) const {
        if (w < 0 || w >= m_dim_w || x < 0 || x >= m_dim_x || y < 0 || y >= m_dim_y || z < 0 || z >= m_dim_z) {
            printf("%d, %d, %d, %d\n", w, x, y, z);
            printf("%d, %d, %d, %d\n", m_dim_w, m_dim_x, m_dim_y, m_dim_z);
            throw std::out_of_range("Matrix4D: Indices out of range.");
        }
        return m_data[w * m_dim_x * m_dim_y * m_dim_z + x * m_dim_y * m_dim_z + y * m_dim_z + z];
    }
#endif

    bool operator==(const Matrix4D<T> &other) const {
        if (m_dim_w != other.m_dim_w || m_dim_x != other.m_dim_x || m_dim_y != other.m_dim_y || m_dim_z != other.m_dim_z) {
            return false;
        }

        for (int w = 0; w < m_dim_w; ++w) {
            for (int x = 0; x < m_dim_x; ++x) {
                for (int y = 0; y < m_dim_y; ++y) {
                    for (int z = 0; z < m_dim_z; ++z) {
                        if ((*this)(w, x, y, z) != other(w, x, y, z)) {
                            return false;
                        }
                    }
                }
            }
        }

        return true;
    }

#if defined(__CUDACC__)
    __host__ __device__ int length() const { return m_dim_w * m_dim_x * m_dim_y * m_dim_z; }
#else
    int length() const { return m_dim_w * m_dim_x * m_dim_y * m_dim_z; }
#endif

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
    T *m_data;
    int m_dim_w, m_dim_x, m_dim_y, m_dim_z;

    // Default constructor
    Matrix4D() { m_data = NULL; }
};

static inline void debug_info(std::string s) {
#ifdef DEBUG
    std::cout << s << std::endl;
#endif
}
#endif
