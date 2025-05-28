#include <hip/hip_runtime.h>
//#include <hipfft.h>
#include <cmath>
#include <iostream>
#include "run_fft_convolution.h"

__global__ void pointwise_complex_multiply(hipfftComplex* a, hipfftComplex* b, hipfftComplex* out, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float a_real = a[idx].x;
        float a_imag = a[idx].y;
        float b_real = b[idx].x;
        float b_imag = b[idx].y;
        out[idx].x = a_real * b_real - a_imag * b_imag;
        out[idx].y = a_real * b_imag + a_imag * b_real;
    }
}

hipError_t run_fft_convolution(float* input, float* filter, float* output,
                                int signal_size, int filter_size, hipStream_t stream) {
    
    return hipSuccess;
}