//
//  param.h
//  metal_cpp
//
//  Created by Derrick on 1/27/24.
//

#ifndef param_h
#define param_h


#endif /* param_h */
typedef struct  {
    unsigned int width1, height1, width2, height2, outputsize, arraySize1, arraySize2;
}matmul_param;

// For customized MatmulInt4 use
typedef struct {
    unsigned int height1;
    unsigned int width3;
    unsigned int width1;
    unsigned int group_size;
} MetalMatMulParams;

// should be inside metal header
// typedef struct {
//     float *A, *C, *scales, *offset;
//     unsigned char *B;
// } MetalMatmulBuffers;
