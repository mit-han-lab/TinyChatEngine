#ifndef COMMON_H
#define COMMON_H
#include <cstdlib>
#include <iostream>
#include <stdexcept>

#include "model.h"

#define DEBUG false
#define USE_OPT_EXP

#define DEBUG_INS(x) \
    if (DEBUG) x

template <typename T>
class Matrix3D {
   public:
    Matrix3D(T* data, int dim_x, int dim_y, int dim_z) : m_data(data), m_dim_x(dim_x), m_dim_y(dim_y), m_dim_z(dim_z) {}

    T& operator()(int x, int y, int z) {
        if (x < 0 || x >= m_dim_x || y < 0 || y >= m_dim_y || z < 0 || z >= m_dim_z) {
            printf("%d, %d, %d\n", x, y, z);
            printf("%d, %d, %d\n", m_dim_x, m_dim_y, m_dim_z);
            throw std::out_of_range("Matrix3D: Indices out of range.");
        }
        return m_data[x * m_dim_y * m_dim_z + y * m_dim_z + z];
    }

    const T& operator()(int x, int y, int z) const {
        if (x < 0 || x >= m_dim_x || y < 0 || y >= m_dim_y || z < 0 || z >= m_dim_z) {
            printf("%d, %d, %d\n", x, y, z);
            printf("%d, %d, %d\n", m_dim_x, m_dim_y, m_dim_z);
            throw std::out_of_range("Matrix3D: Indices out of range.");
        }
        return m_data[x * m_dim_y * m_dim_z + y * m_dim_z + z];
    }

    bool operator==(const Matrix3D<T>& other) const {
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

    int length() const { return m_dim_x * m_dim_y * m_dim_z; }
    int m_dim_x, m_dim_y, m_dim_z;
    T* m_data;

    // Default constructor
    Matrix3D() { m_data = NULL; }
};

static inline void debug_info(std::string s) {
#ifdef DEBUG
    std::cout << s << std::endl;
#endif
}
#endif
