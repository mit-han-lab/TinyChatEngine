#include "common.cuh"
#include <cassert>



class Embedding_half {
   public:
    Embedding_half(int embed_dim_, int voc_size_, int padding_idx_, Matrix3D_cuda<float> lookup_)
        : embed_dim(embed_dim_), voc_size(voc_size_), padding_idx(padding_idx_), lookup(lookup_) {
            assert(lookup_.m_dim_y == voc_size_);
            assert(lookup_.m_dim_z == embed_dim_);
        }
    Embedding_half(){};
    void forward(Matrix3D_cuda<int> input_id, Matrix3D_cuda<float> output);
    int embed_dim, voc_size, padding_idx;
    Matrix3D_cuda<float> lookup;
private:
    std::string profile_name = "Embedding";
};


void load_Embedding_params(Embedding_half &op, std::string prefix);
