#pragma once
#include <hip/hip_runtime.h>

hipError_t run_naive_convolution_2d(const float* input, const float* filter, float* output, 
                                    int input_width_size, int input_height_size, int filter_size);