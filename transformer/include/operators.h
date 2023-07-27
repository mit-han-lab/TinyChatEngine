#ifndef OPERATORS_H
#define OPERATORS_H
#include <cassert>

#include "common.h"
#include "matmul.h"

#define BLK_SIZE 16
#define NUM_THREAD 4

// include all ops
#include "ops/BMM_F32T.h"
#include "ops/BMM_S8T_S8N_F32T.h"
#include "ops/BMM_S8T_S8N_S8T.h"
#include "ops/Embedding.h"
#include "ops/LayerNorm.h"
#include "ops/LayerNormQ.h"
#include "ops/LlamaRMSNorm.h"
#include "ops/RotaryPosEmb.h"
#include "ops/W8A8B8O8Linear.h"
#include "ops/W8A8B8O8LinearReLU.h"
#include "ops/W8A8BFP32OFP32Linear.h"
#include "ops/arg_max.h"
#include "ops/linear.h"

void softmax(const Matrix3D<float> &input, Matrix3D<float> &output, int dim);
void batch_Add(const Matrix3D<float> &input, const Matrix3D<float> &input2, Matrix3D<float> &output);
template <typename T>
void linear(Matrix3D<T> &a, Matrix3D<T> &b, Matrix3D<T> &c);

#endif  // OPERATORS_H
