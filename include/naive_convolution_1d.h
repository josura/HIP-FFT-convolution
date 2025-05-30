#pragma once
#include <hip/hip_runtime.h>


hipError_t run_naive_convolution_1d(const float* input, const float* filter, float* output, 
                                    int signal_size, int filter_size);