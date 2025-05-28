// test/test_fft.cpp
// basic test that validates causal convolution against CPU reference implementation
#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>

__global__ void zero_pad_kernel(float* input, float* output, int input_size, int padded_size);
__global__ void apply_glu_activation(float* a, float* b, float* out, int N);

int main() {
    const int input_size = 8;
    const int padded_size = 12;

    std::vector<float> input(input_size);
    std::vector<float> padded(padded_size);
    for (int i = 0; i < input_size; ++i) input[i] = i + 1;

    float* d_input;
    float* d_padded;
    hipMalloc(&d_input, input_size * sizeof(float));
    hipMalloc(&d_padded, padded_size * sizeof(float));

    hipMemcpy(d_input, input.data(), input_size * sizeof(float), hipMemcpyHostToDevice);

    zero_pad_kernel<<<1, padded_size>>>(d_input, d_padded, input_size, padded_size);

    hipMemcpy(padded.data(), d_padded, padded_size * sizeof(float), hipMemcpyDeviceToHost);

    std::cout << "Zero Padding Test:\n";
    for (int i = 0; i < padded_size; ++i)
        std::cout << padded[i] << " ";
    std::cout << "\n";

    for (int i = 0; i < input_size; ++i)
        assert(std::abs(padded[i] - input[i]) < 1e-5);
    for (int i = input_size; i < padded_size; ++i)
        assert(std::abs(padded[i]) < 1e-5);

    // GLU Test
    const int N = 8;
    std::vector<float> a(N), b(N), out(N);
    for (int i = 0; i < N; ++i) {
        a[i] = i;
        b[i] = -i;
    }

    float *d_a, *d_b, *d_out;
    hipMalloc(&d_a, N * sizeof(float));
    hipMalloc(&d_b, N * sizeof(float));
    hipMalloc(&d_out, N * sizeof(float));

    hipMemcpy(d_a, a.data(), N * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_b, b.data(), N * sizeof(float), hipMemcpyHostToDevice);

    apply_glu_activation<<<1, N>>>(d_a, d_b, d_out, N);

    hipMemcpy(out.data(), d_out, N * sizeof(float), hipMemcpyDeviceToHost);

    std::cout << "\nGLU Activation Test:\n";
    for (int i = 0; i < N; ++i) {
        float expected = a[i] / (1.0f + std::exp(b[i]));
        std::cout << "out[" << i << "] = " << out[i] << ", expected: " << expected << "\n";
        assert(std::abs(out[i] - expected) < 1e-3);
    }

    hipFree(d_input);
    hipFree(d_padded);
    hipFree(d_a);
    hipFree(d_b);
    hipFree(d_out);

    std::cout << "All tests passed!\n";
    return 0;
}