#pragma once

typedef struct {
    unsigned int m; //row1 (only for matmul)
    unsigned int n; //col2 (only for matmul)
    unsigned int k; //col1 (only for matmul)
    unsigned int group_size; // for matmulInt4

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
