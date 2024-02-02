#include <cmath>
#include <iomanip>

#include "operators.h"
#include "utils.h"

void load_Conv2D(Conv2D &op, std::string prefix) {
    read_to_array((prefix + "/weight.bin").c_str(), op.params.weight.m_data, op.params.weight.length());
    // if (has_bias) {
    //     read_to_array((prefix + "/bias.bin").c_str(), op.params.bias.m_data, op.params.bias.length());
    // }
}

// #define MIN(a, b) ((a) < (b) ? (a) : (b))
// #define MAX(a, b) ((a) < (b) ? (b) : (a))
float ActivationFunctionWithMinMax(float x, float output_activation_min, float output_activation_max) {
  return MIN(MAX(x, output_activation_min), output_activation_max);
}

int Offset(const uint16_t dims_data1, const uint16_t dims_data2, const uint16_t dims_data3, int i0, int i1, int i2, int i3) {
  return ((i0 * dims_data1 + i1) * dims_data2 + i2) * dims_data3 + i3;
}

void Conv2D::forward(const Matrix3D<float> &input, Matrix3D<float> &output) {
    PROFILE_START(profile_name);
    // Matrix4D<float> weight = params.weight;
    const float* filter_data = params.weight.m_data;
    // Matrix3D<float> bias = params.bias;
    const float* input_data = input.m_data; 
    float* output_data = output.m_data;
    const int input_depth = input.m_dim_x, input_width = input.m_dim_y, input_height = input.m_dim_z;
    const int filter_input_depth = params.weight.m_dim_w, filter_width = params.weight.m_dim_x, filter_height = params.weight.m_dim_y;
    const int output_depth = output.m_dim_x, output_width = output.m_dim_y, output_height = output.m_dim_z;
    const int batches = 1;

    const int stride_width = params.stride_width, stride_height = params.stride_height;
    const int dilation_width_factor = params.dilation_width_factor, dilation_height_factor = params.dilation_height_factor;
    const int pad_width = params.padding_width, pad_height = params.padding_height;
    const float output_activation_min = params.float_activation_min, output_activation_max = params.float_activation_max;

    // assert(output.m_dim_x == input.m_dim_x);
    // assert(output.m_dim_y == input.m_dim_y);
    // assert(output.m_dim_z == input.m_dim_z);
    // assert(input.m_dim_z == weight.m_dim_z);
    // assert(input.m_dim_z == bias.m_dim_z);

    const int groups = input_depth / filter_input_depth;
    const int filters_per_group = output_depth / groups;

    for (int batch = 0; batch < batches; ++batch) {
        for (int out_y = 0; out_y < output_height; ++out_y) {
            const int in_y_origin = (out_y * stride_height) - pad_height;
            for (int out_x = 0; out_x < output_width; ++out_x) {
                const int in_x_origin = (out_x * stride_width) - pad_width;
                for (int out_channel = 0; out_channel < output_depth; ++out_channel) {
                    int group = out_channel / filters_per_group;
                    float total = 0.f;

                    for (int filter_y = 0; filter_y < filter_height; ++filter_y) {
                        const int in_y = in_y_origin + dilation_height_factor * filter_y;
                        for (int filter_x = 0; filter_x < filter_width; ++filter_x) {
                            const int in_x = in_x_origin + dilation_width_factor * filter_x;

                            // Zero padding by omitting the areas outside the image.
                            const bool is_point_inside_image =
                                (in_x >= 0) && (in_x < input_width) && (in_y >= 0) &&
                                (in_y < input_height);

                            if (!is_point_inside_image) {
                                continue;
                            }
                            for (int in_channel = 0; in_channel < filter_input_depth; ++in_channel) {
                                float input_value =
                                    input_data[Offset(input_height, input_width, input_depth, batch, in_y, in_x,
                                                        in_channel + group * filter_input_depth)];
                                float filter_value = filter_data[Offset(
                                    filter_height, filter_width, input_depth, out_channel, filter_y, filter_x, in_channel)];
                                total += (input_value * filter_value);
                            }
                        }
                    }
                    // float bias_value = 0.0f;
                    // if (has_bias) {
                    //     bias_value = bias_data[out_channel];
                    // }
                    // output_data[Offset(output_height, output_width, output_depth, batch, out_y, out_x, out_channel)] =
                    //     ActivationFunctionWithMinMax(total + bias_value, output_activation_min, output_activation_max);
                    output_data[Offset(output_height, output_width, output_depth, batch, out_y, out_x, out_channel)] = total;
                }
            }
        }
    }

    PROFILE_END(profile_name);
}
