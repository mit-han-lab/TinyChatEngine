#include "ops/Gelu.h"

#include <cassert>

static const float GELU_COEF_A    = 0.044715f;
static const float GELU_QUICK_COEF = -1.702f;
static const float SQRT_2_OVER_PI = 0.79788456080286535587989211986876f;

float Gelu_imp(float x) {
    return 0.5f * x * (1.0f + tanhf(SQRT_2_OVER_PI * x * (1.0f + GELU_COEF_A * x * x)));
}

void Gelu(Matrix3D<float> a) {
    for (int i = 0; i < a.length(); i++) {
        a.m_data[i] = Gelu_imp(a.m_data[i]);
    }
}

float Gelu_quick_imp(float x) {
    return x * (1.0f / (1.0f + expf(GELU_QUICK_COEF * x)));
}

void Gelu_quick(Matrix3D<float> a) {
    for (int i = 0; i < a.length(); i++) {
        a.m_data[i] = Gelu_quick_imp(a.m_data[i]);
    }
}
