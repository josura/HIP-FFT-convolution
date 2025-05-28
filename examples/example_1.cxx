#include <iostream>
#include <vector>
#include <hip/hip_runtime.h>
// #include "conv_layer.hip"
// #include "fft_utils.hip"

// HIP kernel for zero padding
__global__ void pad_kernel(const float* input, float* output, int in_size, int out_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < out_size) {
        if (idx < in_size)
            output[idx] = input[idx];
        else
            output[idx] = 0.0f;
    }
}

// HIP kernel for sigmoid activation
__global__ void sigmoid_kernel(float* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = 1.0f / (1.0f + expf(-data[idx]));
    }
}

int main() {
    const int in_size = 8;
    const int out_size = 16;
    std::vector<float> h_input(in_size, 1.0f); // Example input: all ones
    std::vector<float> h_padded(out_size);

    float *d_input, *d_padded;
    hipMalloc(&d_input, in_size * sizeof(float));
    hipMalloc(&d_padded, out_size * sizeof(float));

    hipMemcpy(d_input, h_input.data(), in_size * sizeof(float), hipMemcpyHostToDevice);

    // Launch padding kernel
    int threads = 256;
    int blocks = (out_size + threads - 1) / threads;
    hipLaunchKernelGGL(pad_kernel, dim3(blocks), dim3(threads), 0, 0, d_input, d_padded, in_size, out_size);

    // Launch sigmoid kernel on padded data
    hipLaunchKernelGGL(sigmoid_kernel, dim3(blocks), dim3(threads), 0, 0, d_padded, out_size);

    hipMemcpy(h_padded.data(), d_padded, out_size * sizeof(float), hipMemcpyDeviceToHost);

    std::cout << "Padded and sigmoid-activated output:\n";
    for (int i = 0; i < out_size; ++i) {
        std::cout << h_padded[i] << " ";
    }
    std::cout << std::endl;

    hipFree(d_input);
    hipFree(d_padded);
    return 0;
}