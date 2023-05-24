#include <cmath>
#include <iomanip>

#include "operators.h"
#include "utils.h"

void load_LayerNormQ(LayerNormQ &op, std::string prefix) {
    read_to_array((prefix + "/weight.bin").c_str(), op.params.weight.m_data, op.params.weight.length());
    read_to_array((prefix + "/bias.bin").c_str(), op.params.bias.m_data, op.params.bias.length());
}

void LayerNormQ::forward(const Matrix3D<float> &x, Matrix3D<int8_t> &output) {
    PROFILE_START(profile_name);
    Matrix3D<float> weight = params.weight;
    Matrix3D<float> bias = params.bias;
    const int last_dims = 2;
    const float eps = 0.00001;

    assert(last_dims == 2);  // support the last dim for now
    assert(output.m_dim_x == x.m_dim_x);
    assert(output.m_dim_y == x.m_dim_y);
    assert(output.m_dim_z == x.m_dim_z);
    assert(x.m_dim_z == weight.m_dim_z);
    assert(x.m_dim_z == bias.m_dim_z);

    for (int i = 0; i < x.m_dim_x; i++) {      // batches
        for (int j = 0; j < x.m_dim_y; j++) {  // samples
            float mean = 0;
            for (int k = 0; k < x.m_dim_z; k++) {  // hideden states
                mean += x(i, j, k);
            }
            mean /= static_cast<float>(x.m_dim_z);
            float squared_diff_sum = 0;
            for (int k = 0; k < x.m_dim_z; k++) {
                float value = static_cast<float>(x(i, j, k));
                squared_diff_sum += (value - mean) * (value - mean);
            }

            float var = squared_diff_sum / static_cast<float>(x.m_dim_z);
            float std_dev = sqrt(var + eps);

            for (int k = 0; k < x.m_dim_z; k++) {
                float value = static_cast<float>(x(i, j, k));
                float fp_out = ((value - mean) / (std_dev) * static_cast<float>(weight(0, 0, k))) +
                               static_cast<float>(bias(0, 0, k));
                output(i, j, k) = static_cast<int8_t>(std::round(fp_out));
                // if (i == 0 && j == 21 && k == 1448){
                //     std::cout << std::setprecision(15) << "mean:" << mean << ",var:" << var << ",std_dev" << std_dev;
                //     std::cout << std::setprecision(15) << ",squared_diff_sum" << squared_diff_sum << ",";
                //     std::cout << std::setprecision(15) << ",value:" << value << ", std_dev:" << std_dev << ",fp_out:"
                //     << fp_out; std::cout << std::setprecision(15) << ",weight:" << weight(0, 0, k) << ",bias:" <<
                //     bias(0, 0, k) << std::endl;
                // }
            }
        }
    }

    // const float c = 1.0 / static_cast<float>(x.m_dim_z);

    // for (int i = 0; i < x.m_dim_x; i++) {      // batches
    //     for (int j = 0; j < x.m_dim_y; j++) {  // samples
    //         float mean = 0;
    //         float sq_sum = 0;
    //         for (int k = 0; k < x.m_dim_z; k++) {  // hidden states
    //             float value = static_cast<float>(x(i, j, k));
    //             mean += value;
    //             sq_sum += value * value;
    //         }
    //         mean *= c;
    //         float rstd_val = std::max(sq_sum * c - mean * mean, 0.0);
    //         rstd_val = 1.0 / std::sqrt(rstd_val + eps);

    //         float scale = rstd_val;
    //         float bias_v = -rstd_val * mean;

    //         for (int k = 0; k < x.m_dim_z; k++) {
    //             float value = static_cast<float>(x(i, j, k));
    //             float gamma_v = static_cast<float>(weight(0, 0, k));
    //             float beta_v = static_cast<float>(bias(0, 0, k));
    //             float fp_out = (value * scale + bias_v) * gamma_v + beta_v;
    //             output(i, j, k) = static_cast<int8_t>(std::round(fp_out));

    //             if (i == 0 && j == 21 && k == 1148){
    //                 std::cout << std::setprecision(15) << "fp_out:" << fp_out;
    //                 std::cout << std::setprecision(15) << ",weight:" << weight(0, 0, k) << ", bias:" <<  bias(0, 0,
    //                 k) << std::endl;
    //             }
    //         }
    //     }
    // }

    PROFILE_END(profile_name);
}
