// benchmark/benchmark_fft.cxx
#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <cassert>
#include "fft_convolution.h"

void benchmark_fft_convolution(int signal_size, int filter_size, int iterations = 100) {
    std::vector<float> signal(signal_size, 1.0f);
    std::vector<float> filter(filter_size, 1.0f);
    std::vector<float> output(signal_size + filter_size - 1, 0.0f);

    float *d_signal, *d_filter, *d_output;
    size_t conv_size = signal_size + filter_size - 1;

    hipMalloc(&d_signal, signal_size * sizeof(float));
    hipMalloc(&d_filter, filter_size * sizeof(float));
    hipMalloc(&d_output, conv_size * sizeof(float));

    hipMemcpy(d_signal, signal.data(), signal_size * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_filter, filter.data(), filter_size * sizeof(float), hipMemcpyHostToDevice);

    hipStream_t stream;
    hipStreamCreate(&stream);

    hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);

    // Warm-up
    run_fft_convolution(d_signal, d_filter, d_output, signal_size, filter_size, stream);
    hipDeviceSynchronize();

    float total_time_ms = 0.0f;
    for (int i = 0; i < iterations; ++i) {
        hipEventRecord(start, stream);

        run_fft_convolution(d_signal, d_filter, d_output, signal_size, filter_size, stream);

        hipEventRecord(stop, stream);
        hipEventSynchronize(stop);

        float milliseconds = 0;
        hipEventElapsedTime(&milliseconds, start, stop);
        total_time_ms += milliseconds;
    }

    std::cout << "[Benchmark] Signal size = " << signal_size
              << ", Filter size = " << filter_size
              << ", Avg Time over " << iterations << " runs: "
              << (total_time_ms / iterations) << " ms" << std::endl;

    hipFree(d_signal);
    hipFree(d_filter);
    hipFree(d_output);
    hipEventDestroy(start);
    hipEventDestroy(stop);
    hipStreamDestroy(stream);
}

int main() {
    std::vector<std::pair<int, int>> test_cases = {
        {128, 3}, {128, 5}, {128, 8}, {512, 5}, {512, 8}, {512, 11}, {1024, 11}, {1024, 15}, {1024, 20}, {1024, 31}, {2048, 31}, {2048, 44}, {2048, 63}, {4096, 63}, {4096, 127}, {8192, 127}, {8192, 255}, {16384, 255}, {16384, 511}, {32768, 511}, {32768, 1023}
    };

    for (const auto& [signal_size, filter_size] : test_cases) {
        benchmark_fft_convolution(signal_size, filter_size);
    }

    return 0;
}
