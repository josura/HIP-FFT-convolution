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
    int conv_size = signal_size + filter_size - 1;
    int fft_size = 1;
    while (fft_size < conv_size) fft_size <<= 1;

    hipfftHandle plan_fwd, plan_inv;
    hipfftComplex *d_input_fft, *d_filter_fft, *d_result_fft;
    float *d_input_padded, *d_filter_padded;

    hipMalloc(&d_input_padded, fft_size * sizeof(float));
    hipMalloc(&d_filter_padded, fft_size * sizeof(float));
    hipMalloc(&d_input_fft, fft_size * sizeof(hipfftComplex));
    hipMalloc(&d_filter_fft, fft_size * sizeof(hipfftComplex));
    hipMalloc(&d_result_fft, fft_size * sizeof(hipfftComplex));

    hipMemsetAsync(d_input_padded, 0, fft_size * sizeof(float), stream);
    hipMemsetAsync(d_filter_padded, 0, fft_size * sizeof(float), stream);

    hipMemcpyAsync(d_input_padded, input, signal_size * sizeof(float), hipMemcpyDeviceToDevice, stream);
    hipMemcpyAsync(d_filter_padded, filter, filter_size * sizeof(float), hipMemcpyDeviceToDevice, stream);

    hipfftPlan1d(&plan_fwd, fft_size, HIPFFT_R2C, 1);
    hipfftPlan1d(&plan_inv, fft_size, HIPFFT_C2R, 1);

    hipfftExecR2C(plan_fwd, d_input_padded, d_input_fft);
    hipfftExecR2C(plan_fwd, d_filter_padded, d_filter_fft);

    int threads = 256;
    int blocks = (fft_size / 2 + 1 + threads - 1) / threads;
    pointwise_complex_multiply<<<blocks, threads, 0, stream>>>(d_input_fft, d_filter_fft, d_result_fft, fft_size / 2 + 1);

    hipfftExecC2R(plan_inv, d_result_fft, d_input_padded);

    hipMemcpyAsync(output, d_input_padded, conv_size * sizeof(float), hipMemcpyDeviceToDevice, stream);

    hipfftDestroy(plan_fwd);
    hipfftDestroy(plan_inv);

    hipFree(d_input_padded);
    hipFree(d_filter_padded);
    hipFree(d_input_fft);
    hipFree(d_filter_fft);
    hipFree(d_result_fft);

    return hipSuccess;
}