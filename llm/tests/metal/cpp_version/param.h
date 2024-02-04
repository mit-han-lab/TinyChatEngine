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
    unsigned int m_dim_x, m_dim_y, m_dim_z;
}matmul_param;

// For customized MatmulInt4 use
typedef struct {
    unsigned int width1, height1, width2, height2, width3, height3, outputsize, arraySize1, arraySize2;
    unsigned int group_size;

    unsigned int m_dim_x, m_dim_y, m_dim_z;
    unsigned int type_size; // for nb
    float eps; // rms_nor
    float scale; // for softmax
    
    int n_past;
    int n_dims;
    int mode;
    int n_orig_ctx;
    float freq_base;
    float freq_scale;
    float ext_factor;
    float attn_factor;
    float beta_fast;
    float beta_slow;
} MetalMatMulParams;

// should be inside metal header
