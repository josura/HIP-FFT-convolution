#include <hip/hip_runtime.h>

__global__ void zero_pad_kernel(float* input, float* output, int input_size, int padded_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < padded_size) {
        if (idx < input_size) {
            output[idx] = input[idx];
        } else {
            output[idx] = 0.0f;
        }
    }
}