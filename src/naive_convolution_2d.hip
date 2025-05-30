#include <hip/hip_runtime.h>

// Naive 2D convolution kernel (single-channel, zero-padding)
__global__ void naive_convolution_2d(
    const float* __restrict__ input,
    const float* __restrict__ kernel,
    float* output,
    int width,
    int height,
    int ksize)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int khalf = ksize / 2;

    if (x < width && y < height) {
        float sum = 0.0f;
        for (int ky = 0; ky < ksize; ++ky) {
            for (int kx = 0; kx < ksize; ++kx) {
                int ix = x + kx - khalf;
                int iy = y + ky - khalf;
                float val = 0.0f;
                if (ix >= 0 && ix < width && iy >= 0 && iy < height) {
                    val = input[iy * width + ix];
                }
                sum += val * kernel[ky * ksize + kx];
            }
        }
        output[y * width + x] = sum;
    }
}


/**
 * @brief Function to perform convolution of a signal with a filter.
 * @param input Pointer to the input array to be convolved(flattened 2D array).
 * @param filter Pointer to the filter array.
 * @param output Pointer to the output array where the result will be stored.
 * @param input_width_size Width of the input array.
 * @param input_height_size Height of the input array.
 * @param filter_size Size of the filter array.
 * @return hipError_t indicating success or failure of the operation.
 * @note This function assumes that the arrays are flattened 2D arrays, meaning that the input, filter, and output are treated as 1D arrays.
 */
hipError_t run_naive_convolution_2d(const float* input, const float* filter, float* output, int input_width_size, int input_height_size, int filter_size) {
    // Define grid and block dimensions
    dim3 blockDim(16, 16);
    //dim3 gridDim((input_size + blockDim.x - 1) / blockDim.x, (input_size + blockDim.y - 1) / blockDim.y);
    int width = input_width_size;  // Assuming square input for simplicity
    int height = input_height_size; // Assuming square input for simplicity
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y);


    // Launch the convolution kernel
    naive_convolution_2d<<<gridDim, blockDim>>>(input, filter, output, input_width_size, input_height_size, filter_size);

    // Return the result of the kernel launch
    return hipGetLastError();
}