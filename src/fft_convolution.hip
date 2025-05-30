/**
 * @file fft_convolution.hip
 * @brief This file contains the implementation of FFT-based convolution using HIP.
 * It includes the necessary kernel for pointwise multiplication of complex numbers and the main function to run the FFT convolution.
 * This implementation uses the HIPFFT library for performing Fast Fourier Transforms.
 * @todo Use the distributed FFT library for better performance. example in here @link https://rocmdocs.amd.com/projects/hipFFT/en/develop/conceptual/overview.html @link
 */
#include <hip/hip_runtime.h>
//#include <hipfft.h>
#include <cmath>
#include <iostream>
#include "fft_convolution.h"

/**
 * @brief HIP kernel performing pointwise multiplication of two complex arrays.
 * @param a Pointer to the first complex array.
 * @param b Pointer to the second complex array.
 * @param out Pointer to the output complex array.
 * @param N Size of the complex arrays.
 * @note This kernel assumes that the input arrays are of size N and that the output array is also of size N.
 * @note The complex numbers are represented as hipfftComplex, which contains two floats: x (real part) and y (imaginary part).
 */
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

/**
 * @brief HIP kernel to normalize the output of the inverse FFT.
 * @param data Pointer to the output data array.
 * @param size Size of the output data array.
 * @param scale Normalization scale factor.
 * @note This kernel divides each element in the output data by the scale factor.
 */
__global__ void normalize_output(float* data, int size, float scale) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
        data[idx] /= scale;
}

/**
 * @brief Function to perform FFT-based convolution of a signal with a filter.
 * @param input Pointer to the input signal array on the device.
 * @param filter Pointer to the filter array on the device.
 * @param output Pointer to the output array on the device.
 * @param signal_size Size of the input signal.
 * @param filter_size Size of the filter.
 * @param stream HIP stream for asynchronous execution.
 * @return hipError_t indicating success or failure of the operation.
 * @todo Use the distributed FFT library for better performance. example in here @link https://rocmdocs.amd.com/projects/hipFFT/en/develop/conceptual/overview.html @link
 * @todo handle errors more gracefully, possibly with exceptions or error codes.
 */
hipError_t run_fft_convolution(float* input, float* filter, float* output,
                                int signal_size, int filter_size, hipStream_t stream) {
    int conv_size = signal_size + filter_size - 1; // considering the padded size for convolution, since it's required for FFT linear convolution
    int fft_size = 1;
    while (fft_size < conv_size) fft_size <<= 1; // Bit shifting to get the next power of two greater than or equal to conv_size, ensuring efficient FFT computation.

    hipfftComplex *d_input_fft, *d_filter_fft, *d_result_fft;
    float *d_input_padded, *d_filter_padded;
    hipError_t err;
    err = hipMalloc(&d_input_padded, fft_size * sizeof(float));
    if (err != hipSuccess) {
        std::cerr << "hipMalloc for d_input_padded failed: " << hipGetErrorString(err) << std::endl;
        return err;
    }
    err = hipMalloc(&d_filter_padded, fft_size * sizeof(float));
    if (err != hipSuccess) {
        std::cerr << "hipMalloc for d_filter_padded failed: " << hipGetErrorString(err) << std::endl;
        err = hipFree(d_input_padded);
        return err;
    }
    err = hipMalloc(&d_input_fft, fft_size * sizeof(hipfftComplex));
    if (err != hipSuccess) {
        std::cerr << "hipMalloc for d_input_fft failed: " << hipGetErrorString(err) << std::endl;
        err = hipFree(d_input_padded);
        err = hipFree(d_filter_padded);
        return err;
    }
    err = hipMalloc(&d_filter_fft, fft_size * sizeof(hipfftComplex));
    if (err != hipSuccess) {
        std::cerr << "hipMalloc for d_filter_fft failed: " << hipGetErrorString(err) << std::endl;
        err = hipFree(d_input_padded);
        err = hipFree(d_filter_padded);
        err = hipFree(d_input_fft);
        return err;
    }
    err = hipMalloc(&d_result_fft, fft_size * sizeof(hipfftComplex));
    if (err != hipSuccess) {
        std::cerr << "hipMalloc for d_result_fft failed: " << hipGetErrorString(err) << std::endl;
        err = hipFree(d_input_padded);
        err = hipFree(d_filter_padded);
        err = hipFree(d_input_fft);
        err = hipFree(d_filter_fft);
        return err;
    }

    err = hipMemsetAsync(d_input_padded, 0, fft_size * sizeof(float), stream); // initial padding
    if (err != hipSuccess) {
        std::cerr << "hipMemsetAsync for d_input_padded failed: " << hipGetErrorString(err) << std::endl;
        err = hipFree(d_input_padded);
        err = hipFree(d_filter_padded);
        err = hipFree(d_input_fft);
        err = hipFree(d_filter_fft);
        err = hipFree(d_result_fft);
        return err;
    }
    err = hipMemsetAsync(d_filter_padded, 0, fft_size * sizeof(float), stream); // initial padding(to filter)
    if (err != hipSuccess) {
        std::cerr << "hipMemsetAsync for d_filter_padded failed: " << hipGetErrorString(err) << std::endl;
        err = hipFree(d_input_padded);
        err = hipFree(d_filter_padded);
        err = hipFree(d_input_fft);
        err = hipFree(d_filter_fft);
        err = hipFree(d_result_fft);
        return err;
    }

    err = hipMemcpyAsync(d_input_padded, input, signal_size * sizeof(float), hipMemcpyDeviceToDevice, stream);
    if (err != hipSuccess) {
        std::cerr << "hipMemcpyAsync for d_input_padded failed: " << hipGetErrorString(err) << std::endl;
        err = hipFree(d_input_padded);
        err = hipFree(d_filter_padded);
        err = hipFree(d_input_fft);
        err = hipFree(d_filter_fft);
        err = hipFree(d_result_fft);
        return err;
    }
    err = hipMemcpyAsync(d_filter_padded, filter, filter_size * sizeof(float), hipMemcpyDeviceToDevice, stream);
    if (err != hipSuccess) {
        std::cerr << "hipMemcpyAsync for d_filter_padded failed: " << hipGetErrorString(err) << std::endl;
        err = hipFree(d_input_padded);
        err = hipFree(d_filter_padded);
        err = hipFree(d_input_fft);
        err = hipFree(d_filter_fft);
        err = hipFree(d_result_fft);
        return err;
    }

    hipfftHandle plan_fwd, plan_inv; // Handles for forward and inverse FFT plans
    hipfftPlan1d(&plan_fwd, fft_size, HIPFFT_R2C, 1); // Create a plan for forward FFT (Real to Complex)
    hipfftPlan1d(&plan_inv, fft_size, HIPFFT_C2R, 1); // Create a plan for inverse FFT (Complex to Real)

    hipfftExecR2C(plan_fwd, d_input_padded, d_input_fft);
    hipfftExecR2C(plan_fwd, d_filter_padded, d_filter_fft);

    int threads = 256;
    int blocks = (fft_size / 2 + 1 + threads - 1) / threads;
    pointwise_complex_multiply<<<blocks, threads, 0, stream>>>(d_input_fft, d_filter_fft, d_result_fft, fft_size / 2 + 1);

    hipfftExecC2R(plan_inv, d_result_fft, d_input_padded);

    // Normalize the real output after inverse FFT
    int norm_threads = 256;
    int norm_blocks = (fft_size + norm_threads - 1) / norm_threads;
    normalize_output<<<norm_blocks, norm_threads, 0, stream>>>(d_input_padded, fft_size, static_cast<float>(fft_size));


    err = hipMemcpyAsync(output, d_input_padded, conv_size * sizeof(float), hipMemcpyDeviceToDevice, stream);
    if (err != hipSuccess) {
        std::cerr << "hipMemcpyAsync for output failed: " << hipGetErrorString(err) << std::endl;
        err = hipFree(d_input_padded);
        err = hipFree(d_filter_padded);
        err = hipFree(d_input_fft);
        err = hipFree(d_filter_fft);
        err = hipFree(d_result_fft);
        return err;
    }

    hipfftDestroy(plan_fwd);
    hipfftDestroy(plan_inv);

    err = hipFree(d_input_padded);
    if (err != hipSuccess) {
        std::cerr << "[WARNING] hipFree for d_input_padded failed: " << hipGetErrorString(err) << std::endl;
    }
    err = hipFree(d_filter_padded);
    if (err != hipSuccess) {
        std::cerr << "[WARNING] hipFree for d_filter_padded failed: " << hipGetErrorString(err) << std::endl;
    }
    err = hipFree(d_input_fft);
    if (err != hipSuccess) {
        std::cerr << "[WARNING] hipFree for d_input_fft failed: " << hipGetErrorString(err) << std::endl;
    }
    err = hipFree(d_filter_fft);
    if (err != hipSuccess) {
        std::cerr << "[WARNING] hipFree for d_filter_fft failed: " << hipGetErrorString(err) << std::endl;
    }
    err = hipFree(d_result_fft);
    if (err != hipSuccess) {
        std::cerr << "[WARNING] hipFree for d_result_fft failed: " << hipGetErrorString(err) << std::endl;
    }

    return hipSuccess;
}