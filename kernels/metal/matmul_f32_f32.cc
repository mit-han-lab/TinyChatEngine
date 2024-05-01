#include "metal_compute.h"
namespace matmul {
    void mat_mul_f32_f32_metal(const struct matmul_params *params){
        struct metal_cgraph *graph = new (struct metal_cgraph);
        graph->n_nodes = 1;
        graph->mm_nodes[0] = (const metal_params *) params;
        metal_graph_compute(METAL_KERNEL_MUL_MM_F32_F32, graph);
    }
}