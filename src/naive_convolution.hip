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