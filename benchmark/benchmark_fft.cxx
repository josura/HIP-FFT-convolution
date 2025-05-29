// benchmark/benchmark_fft.cxx
#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <cassert>
#include "fft_convolution.h"

// Function to run FFT convolution in cpu (single threaded)
void run_fft_convolution_cpu(const std::vector<float>& signal, const std::vector<float>& filter, std::vector<float>& output) {
    int signal_size = signal.size();
    int filter_size = filter.size();
    int conv_size = signal_size + filter_size - 1;

    // Initialize output to zero
    std::fill(output.begin(), output.end(), 0.0f);

    // Perform convolution
    for (int i = 0; i < signal_size; ++i) {
        for (int j = 0; j < filter_size; ++j) {
            if (i + j < conv_size) {
                output[i + j] += signal[i] * filter[j];
            }
        }
    }
}

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
    float total_time_cpu_ms = 0.0f;
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
        err = hipEventElapsedTime(&milliseconds, start, stop);
        if (err != hipSuccess) {
            std::cerr << "Error calculating elapsed time for iteration " << i << ": " << hipGetErrorString(err) << std::endl;
            break;
        }
        total_time_ms += milliseconds;
        // Run CPU convolution for comparison
        auto start_cpu = std::chrono::high_resolution_clock::now();
        run_fft_convolution_cpu(signal, filter, output);
        auto end_cpu = std::chrono::high_resolution_clock::now();
        std::chrono::duration<float, std::milli> cpu_duration = end_cpu - start_cpu;
        total_time_cpu_ms += cpu_duration.count();
    }

    std::cout << "[Benchmark] Signal size = " << signal_size
              << ", Filter size = " << filter_size
              << ", Avg Time over " << iterations << " runs: "
              << (total_time_ms / iterations) << " ms" << std::endl;
    std::cout << "[Benchmark] CPU Avg Time over " << iterations << " runs: "
                << (total_time_cpu_ms / iterations) << " ms" << std::endl;
    std::cout << "[Benchmark] Speedup: "
              << (total_time_cpu_ms / total_time_ms) << "x" << std::endl;

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
    std::cout << "Starting FFT Convolution Benchmark..." << std::endl;
    std::cout << "Smallest signal size: 128, largest signal size: 32768" << std::endl;
    std::vector<std::pair<int, int>> test_cases = {
        {128, 3}, {128, 5}, {128, 8}, {512, 5}, {512, 8}, {512, 11}, {1024, 11}, {1024, 15}, {1024, 20}, {1024, 31}, {2048, 31}, {2048, 44}, {2048, 63}, {4096, 63}, {4096, 127}, {8192, 127}, {8192, 255}, {16384, 255}, {16384, 511}, {32768, 511}, {32768, 1023}
    };

    for (const auto& [signal_size, filter_size] : test_cases) {
        benchmark_fft_convolution(signal_size, filter_size);
    }
    // Additional test cases for larger sizes
    std::cout << "Additional test cases for larger sizes..." << std::endl;
    std::vector<std::pair<int, int>> additional_cases = {
        {65536, 1023}, {65536, 2047}, {131072, 2047}, {131072, 4095}, {262144, 4095}, {262144, 8191}, {524288, 8191}, {524288, 16383}, {1048576, 16383}, {1048576, 32767}, {2097152, 32767}, {2097152, 65535}
    };
    for (const auto& [signal_size, filter_size] : additional_cases) {
        benchmark_fft_convolution(signal_size, filter_size);
    }
    // Very large test cases
    std::cout << "Very large test cases..." << std::endl;
    std::vector<std::pair<int, int>> very_large_cases = {
        {41943040, 65535}, {41943040, 131071}, {83886080, 131071}, {83886080, 262143}, {167772160, 262143}, {167772160, 524287}, {335544320, 524287}, {335544320, 1048575}
    };
    for (const auto& [signal_size, filter_size] : very_large_cases) {
        benchmark_fft_convolution(signal_size, filter_size);
    }

    std::cout << "All benchmarks completed successfully." << std::endl;

    return 0;
}
