#pragma once
#include <hip/hip_runtime.h>
#include <hipfft/hipfft.h>

hipError_t run_fft_convolution(float* input, float* filter, float* output,
                                int signal_size, int filter_size, hipStream_t stream);
