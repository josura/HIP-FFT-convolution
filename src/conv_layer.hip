#include <hip/hip_runtime.h>

__global__ void apply_glu_activation(float* a, float* b, float* out, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float sigmoid = 1.0f / (1.0f + expf(-b[idx]));
        out[idx] = a[idx] * sigmoid;
    }
}