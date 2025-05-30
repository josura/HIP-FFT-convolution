#include <hip/hip_runtime.h>

__global__ void naive_convolution_1d(
    const float* __restrict__ input,
    const float* __restrict__ kernel,
    float* output,
    int input_size,
    int kernel_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int conv_size = input_size + kernel_size - 1;
    if (idx < conv_size) {
        float sum = 0.0f;
        for (int k = 0; k < kernel_size; ++k) {
            int input_idx = idx - k;
            if (input_idx >= 0 && input_idx < input_size) {
                sum += input[input_idx] * kernel[k];
            }
        }
        output[idx] = sum;
    }
}

/**
 * @brief Runs a naive 1D convolution on the GPU.
 */
hipError_t run_naive_convolution_1d(const float* input, const float* kernel, float* output, 
                                    int input_size, int kernel_size)
{
    dim3 grid_dim((input_size + kernel_size - 1 + 255) / 256); // Calculate grid size based on input size and kernel size
    dim3 block_dim(256); // Use a block size of 256 threads
    hipError_t err = hipSuccess;
    
    // Launch the kernel
    naive_convolution_1d<<<grid_dim, block_dim>>>(input, kernel, output, input_size, kernel_size);
    
    // Check for errors
    err = hipGetLastError();
    if (err != hipSuccess) {
        return err;
    }
    
    return hipDeviceSynchronize();
}