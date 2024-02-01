#pragma once

typedef struct {
    unsigned int m; //row1
    unsigned int n; //col2
    unsigned int k; //col1
    unsigned int group_size;
    unsigned int type_size;
    float eps;
} MetalMatMulParams;
