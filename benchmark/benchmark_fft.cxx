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

    hipError_t err;
    err = hipMalloc(&d_signal, signal_size * sizeof(float));
    if (err != hipSuccess) {
        std::cerr << "Error allocating device memory for signal with (signal,filter) size ("<< signal_size << "," << filter_size << ") : " << hipGetErrorString(err) << std::endl;
        return;
    }
    err = hipMalloc(&d_filter, filter_size * sizeof(float));
    if (err != hipSuccess) {
        std::cerr << "Error allocating device memory for filter with (signal,filter) size ("<< signal_size << "," << filter_size << ") : " << hipGetErrorString(err) << std::endl;
        err = hipFree(d_signal);
        return;
    }
    err = hipMalloc(&d_output, conv_size * sizeof(float));
    if (err != hipSuccess) {
        std::cerr << "Error allocating device memory for output with (signal,filter) size ("<< signal_size << "," << filter_size << ") : " << hipGetErrorString(err) << std::endl;
        err = hipFree(d_signal);
        err = hipFree(d_filter);
        return;
    }

    err = hipMemcpy(d_signal, signal.data(), signal_size * sizeof(float), hipMemcpyHostToDevice);
    if (err != hipSuccess) {
        std::cerr << "Error copying signal data to device with (signal,filter) size ("<< signal_size << "," << filter_size << ") : " << hipGetErrorString(err) << std::endl;
        err = hipFree(d_signal);
        err = hipFree(d_filter);
        err = hipFree(d_output);
        return;
    }
    err = hipMemcpy(d_filter, filter.data(), filter_size * sizeof(float), hipMemcpyHostToDevice);
    if (err != hipSuccess) {
        std::cerr << "Error copying filter data to device with (signal,filter) size ("<< signal_size << "," << filter_size << ") : " << hipGetErrorString(err) << std::endl;
        err = hipFree(d_signal);
        err = hipFree(d_filter);
        err = hipFree(d_output);
        return;
    }

    hipStream_t stream;
    err = hipStreamCreate(&stream);
    if (err != hipSuccess) {
        std::cerr << "Error creating HIP stream with (signal,filter) size ("<< signal_size << "," << filter_size << ") : " << hipGetErrorString(err) << std::endl;
        err = hipFree(d_signal);
        err = hipFree(d_filter);
        err = hipFree(d_output);
        return;
    }

    hipEvent_t start, stop;
    err = hipEventCreate(&start);
    if (err != hipSuccess) {
        std::cerr << "Error creating start event with (signal,filter) size ("<< signal_size << "," << filter_size << ") : " << hipGetErrorString(err) << std::endl;
        err = hipFree(d_signal);
        err = hipFree(d_filter);
        err = hipFree(d_output);
        err = hipStreamDestroy(stream);
        return;
    }
    err = hipEventCreate(&stop);
    if (err != hipSuccess) {
        std::cerr << "Error creating stop event with (signal,filter) size ("<< signal_size << "," << filter_size << ") : " << hipGetErrorString(err) << std::endl;
        err = hipFree(d_signal);
        err = hipFree(d_filter);
        err = hipFree(d_output);
        err = hipEventDestroy(start);
        err = hipStreamDestroy(stream);
        return;
    }

    // Warm-up
    err = run_fft_convolution(d_signal, d_filter, d_output, signal_size, filter_size, stream);
    if (err != hipSuccess) {
        std::cerr << "Error running FFT convolution during warm-up with (signal,filter) size ("<< signal_size << "," << filter_size << ") : " << hipGetErrorString(err) << std::endl;
        err = hipFree(d_signal);
        err = hipFree(d_filter);
        err = hipFree(d_output);
        err = hipEventDestroy(start);
        err = hipEventDestroy(stop);
        err = hipStreamDestroy(stream);
        return;
    }
    err = hipDeviceSynchronize();
    if (err != hipSuccess) {
        std::cerr << "Error synchronizing device after warm-up with (signal,filter) size ("<< signal_size << "," << filter_size << ") : " << hipGetErrorString(err) << std::endl;
        err = hipFree(d_signal);
        err = hipFree(d_filter);
        err = hipFree(d_output);
        err = hipEventDestroy(start);
        err = hipEventDestroy(stop);
        err = hipStreamDestroy(stream);
        return;
    }

    // Benchmarking
    float total_time_ms = 0.0f;
    for (int i = 0; i < iterations; ++i) {
        err = hipEventRecord(start, stream);
        if (err != hipSuccess) {
            std::cerr << "Error recording start event for iteration "<< i << " : " << hipGetErrorString(err) << std::endl;
            break;
        }

        err = run_fft_convolution(d_signal, d_filter, d_output, signal_size, filter_size, stream);
        if (err != hipSuccess) {
            std::cerr << "Error running FFT convolution for iteration " << i << ": " << hipGetErrorString(err) << std::endl;
            break;
        }

        err = hipEventRecord(stop, stream);
        if (err != hipSuccess) {
            std::cerr << "Error recording stop event for iteration " << i << ": " << hipGetErrorString(err) << std::endl;
            break;
        }
        err = hipEventSynchronize(stop);
        if (err != hipSuccess) {
            std::cerr << "Error synchronizing stop event for iteration " << i << ": " << hipGetErrorString(err) << std::endl;
            break;
        }

        float milliseconds = 0;
        hipEventElapsedTime(&milliseconds, start, stop);
        total_time_ms += milliseconds;
    }

    std::cout << "[Benchmark] Signal size = " << signal_size
              << ", Filter size = " << filter_size
              << ", Avg Time over " << iterations << " runs: "
              << (total_time_ms / iterations) << " ms" << std::endl;

    err = hipFree(d_signal);
    if (err != hipSuccess) {
        std::cerr << "Error freeing device memory for signal with (signal,filter) size ("<< signal_size << "," << filter_size << ") : " << hipGetErrorString(err) << std::endl;
    }
    err = hipFree(d_filter);
    err = hipFree(d_output);
    err = hipEventDestroy(start);
    err = hipEventDestroy(stop);
    err = hipStreamDestroy(stream);
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
