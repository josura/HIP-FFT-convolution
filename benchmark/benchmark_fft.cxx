// benchmark/benchmark_fft.cxx
#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <cassert>
#include "fft_convolution.h"
#include "naive_convolution_1d.h"
#include "naive_convolution_2d.h"

// Function to run convolution in cpu (single threaded)
void run_convolution_cpu(const std::vector<float>& signal, const std::vector<float>& filter, std::vector<float>& output) {
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

void benchmark_fft_convolution(int signal_size, int filter_size, int iterations = 100, bool no_gpu = false, bool no_cpu = false) {
    std::vector<float> signal(signal_size, 1.0f);
    std::vector<float> filter(filter_size, 1.0f);
    std::vector<float> output(signal_size + filter_size - 1, 0.0f);
    std::vector<float> signal_classic_convolution(signal_size, 1.0f);
    std::vector<float> filter_classic_convolution(filter_size, 1.0f);
    std::vector<float> output_classic_convolution(signal_size + filter_size - 1, 0.0f);


    float total_time_ms = 0.0f;
    float total_time_classic_ms = 0.0f;
    float total_time_cpu_ms = 0.0f;

    //GPU
    if(!no_gpu) {
        hipError_t err;

        // Classic convolution
        float *d_signal_classic, *d_filter_classic, *d_output_classic;
        size_t conv_size_classic = signal_size + filter_size - 1;
        err = hipMalloc(&d_signal_classic, signal_size * sizeof(float));
        if (err != hipSuccess) {
            std::cerr << "Error allocating device memory for signal with (signal,filter) size ("<< signal_size << "," << filter_size << ") : " << hipGetErrorString(err) << std::endl;
            return;
        }
        err = hipMalloc(&d_filter_classic, filter_size * sizeof(float));
        if (err != hipSuccess) {
            std::cerr << "Error allocating device memory for filter with (signal,filter) size ("<< signal_size << "," << filter_size << ") : " << hipGetErrorString(err) << std::endl;
            err = hipFree(d_signal_classic);
            return;
        }
        err = hipMalloc(&d_output_classic, conv_size_classic * sizeof(float));
        if (err != hipSuccess) {
            std::cerr << "Error allocating device memory for output with (signal,filter) size ("<< signal_size << "," << filter_size << ") : " << hipGetErrorString(err) << std::endl;
            err = hipFree(d_signal_classic);
            err = hipFree(d_filter_classic);
            return;
        }
        err = hipMemcpy(d_signal_classic, signal_classic_convolution.data(), signal_size * sizeof(float), hipMemcpyHostToDevice);
        if (err != hipSuccess) {
            std::cerr << "Error copying signal data to device with (signal,filter) size ("<< signal_size << "," << filter_size << ") : " << hipGetErrorString(err) << std::endl;
            err = hipFree(d_signal_classic);
            err = hipFree(d_filter_classic);
            err = hipFree(d_output_classic);
            return;
        }
        err = hipMemcpy(d_filter_classic, filter_classic_convolution.data(), filter_size * sizeof(float), hipMemcpyHostToDevice);
        if (err != hipSuccess) {
            std::cerr << "Error copying filter data to device with (signal,filter) size ("<< signal_size << "," << filter_size << ") : " << hipGetErrorString(err) << std::endl;
            err = hipFree(d_signal_classic);
            err = hipFree(d_filter_classic);
            err = hipFree(d_output_classic);
            return;
        }
        // FFT convolution
        float *d_signal, *d_filter, *d_output;
        size_t conv_size = signal_size + filter_size - 1;

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

        // Create events for timing
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
            // Run classic convolution for comparison
            err = hipEventRecord(start, stream);
            if (err != hipSuccess) {
                std::cerr << "Error recording start event for classic convolution in iteration " << i << ": " << hipGetErrorString(err) << std::endl;
                break;
            }
            err = run_naive_convolution_1d(d_signal_classic, d_filter_classic, d_output_classic, signal_size, filter_size);
            //err = run_naive_convolution_2d(d_signal_classic, d_filter_classic, d_output_classic, signal_size, signal_size, filter_size);
            if (err != hipSuccess) {
                std::cerr << "Error running classic convolution for iteration " << i << ": " << hipGetErrorString(err) << std::endl;
                break;
            }
            err = hipEventRecord(stop, stream);
            if (err != hipSuccess) {
                std::cerr << "Error recording stop event for classic convolution in iteration " << i << ": " << hipGetErrorString(err) << std::endl;
                break;
            }
            err = hipEventSynchronize(stop);
            if (err != hipSuccess) {
                std::cerr << "Error synchronizing stop event for classic convolution in iteration " << i << ": " << hipGetErrorString(err) << std::endl;
                break;
            }
            float classic_milliseconds = 0;
            err = hipEventElapsedTime(&classic_milliseconds, start, stop);
            if (err != hipSuccess) {
                std::cerr << "Error calculating elapsed time for classic convolution in iteration " << i << ": " << hipGetErrorString(err) << std::endl;
                break;
            }
            total_time_classic_ms += classic_milliseconds;
        }

        // Cleanup

        err = hipFree(d_signal);
        err = hipFree(d_filter);
        err = hipFree(d_output);
        err = hipEventDestroy(start);
        err = hipEventDestroy(stop);
        err = hipStreamDestroy(stream);
    }

    //CPU
    if(!no_cpu) {
        auto start_cpu = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iterations; ++i) {
            run_convolution_cpu(signal, filter, output);
        }
        auto end_cpu = std::chrono::high_resolution_clock::now();
        std::chrono::duration<float, std::milli> cpu_duration = end_cpu - start_cpu;
        total_time_cpu_ms = cpu_duration.count() / iterations;
    }

    // Printing results section
    if(!no_gpu){
        std::cout << "[Benchmark GPU] Signal size = " << signal_size
              << ", Filter size = " << filter_size
              << ", Avg Time over " << iterations << " runs: "
              << (total_time_ms / iterations) << " ms" << std::endl;
        std::cout << "[Benchmark GPU] Classic Convolution Avg Time over " << iterations << " runs: "
                << (total_time_classic_ms / iterations) << " ms" << std::endl;
        std::cout << "[Benchmark GPU] Classic Convolution Speedup: "
                << (total_time_classic_ms / total_time_ms) << "x" << std::endl;
    
    }
    if(!no_cpu){
        std::cout << "[Benchmark CPU] CPU Avg Time over " << iterations << " runs for (signal,filter) size ("<< signal_size << "," << filter_size << "): "
                << (total_time_cpu_ms / iterations) << " ms" << std::endl;
    }
    if(!no_gpu && !no_cpu) {
        std::cout << "[Benchmark CPU vs GPU] CPU Speedup: "
              << (total_time_cpu_ms / total_time_ms) << "x" << std::endl;
    }
}

int main(int argc, char** argv) {
    // Control if the command line arguments have the following options:
    // --help: Print help message
    for(int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--help") {
            std::cout << "Usage: " << argv[0] << " [--help]" << std::endl;
            std::cout << "This program benchmarks FFT convolution on HIP devices." << std::endl;
            // TODO add the other options here
            return 0;
        }
    }
    // --mode: select the mode of the benchmark. Available modes are: all, gpu, cpu. Default is all.
    bool no_cpu = false;
    bool no_gpu = false;
    for(int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--mode") {
            if (i + 1 < argc) {
                std::string mode = argv[++i];
                if (mode == "cpu") {
                    no_gpu = true;
                } else if (mode == "gpu") {
                    no_cpu = true;
                } else if (mode == "all") {
                    no_cpu = false;
                    no_gpu = false;
                } else {
                    std::cerr << "Unknown mode: " << mode << ". Available modes are: all, gpu, cpu. Default is all" << std::endl;
                    return -1;
                }
            } else {
                std::cerr << "--mode option requires an argument." << std::endl;
                return -1;
            }
        }
    }

    std::cout << "Starting FFT Convolution Benchmark..." << std::endl;
    // Print the device properties
    int device_count;
    hipError_t err = hipGetDeviceCount(&device_count);
    if (err != hipSuccess) {
        std::cerr << "Error getting device count: " << hipGetErrorString(err) << std::endl;
        return -1;
    }
    std::cout << "Number of HIP devices: " << device_count << std::endl;
    if (device_count == 0) {
        std::cerr << "No HIP devices found." << std::endl;
        return -1;
    }
    int device_id = 0; // Use the first device
    err = hipSetDevice(device_id);
    if (err != hipSuccess) {
        std::cerr << "Error setting device: " << hipGetErrorString(err) << std::endl;
        return -1;
    }
    hipDeviceProp_t device_prop;
    err = hipGetDeviceProperties(&device_prop, device_id);
    if (err != hipSuccess) {
        std::cerr << "Error getting device properties: " << hipGetErrorString(err) << std::endl;
        return -1;
    }
    std::cout << "Device Name: " << device_prop.name << std::endl;
    std::cout << "Total Global Memory: " << device_prop.totalGlobalMem / (1024 * 1024) << " MB" << std::endl;
    std::cout << "Max Threads Per Block: " << device_prop.maxThreadsPerBlock << std::endl;
    std::cout << "Max Threads Dim: (" << device_prop.maxThreadsDim[0] << ", "
              << device_prop.maxThreadsDim[1] << ", "
              << device_prop.maxThreadsDim[2] << ")" << std::endl;
    std::cout << "Max Grid Size: (" << device_prop.maxGridSize[0] << ", "
              << device_prop.maxGridSize[1] << ", "
              << device_prop.maxGridSize[2] << ")" << std::endl;
    std::cout << "Max Shared Memory Per Block: " << device_prop.sharedMemPerBlock / 1024 << " KB" << std::endl;
    std::cout << "Max Memory Pitch: " << device_prop.memPitch / 1024 << " KB" << std::endl;
    std::cout << "Max Registers Per Block: " << device_prop.regsPerBlock << std::endl;
    std::cout << "Warp Size: " << device_prop.warpSize << std::endl;
    std::cout << "Clock Rate: " << device_prop.clockRate / 1000 << " MHz" << std::endl;
    std::cout << "Total Constant Memory: " << device_prop.totalConstMem / 1024 << " KB" << std::endl;
    std::cout << "Compute Capability: " << device_prop.major << "." << device_prop.minor << std::endl;
    std::cout << "Concurrent Kernels: " << (device_prop.concurrentKernels ? "Yes" : "No") << std::endl;
    std::cout << "Unified Addressing: " << (device_prop.unifiedAddressing ? "Yes" : "No") << std::endl;
    std::cout << "L2 Cache Size: " << device_prop.l2CacheSize / 1024 << " KB" << std::endl;
    std::cout << "Max Threads Per Multiprocessor: " << device_prop.maxThreadsPerMultiProcessor << std::endl;
    std::cout << "Multi-Processor Count: " << device_prop.multiProcessorCount << std::endl;
    std::cout << "Memory Clock Rate: " << device_prop.memoryClockRate / 1000 << " MHz" << std::endl;
    std::cout << "Memory Bus Width: " << device_prop.memoryBusWidth << " bits" << std::endl;
    std::cout << "Total Memory: " << device_prop.totalGlobalMem / (1024 * 1024) << " MB" << std::endl;
    std::cout << "Max Texture Dimension 1D: " << device_prop.maxTexture1D << std::endl;
    std::cout << "Max Texture Dimension 2D: (" << device_prop.maxTexture2D[0] << ", "
              << device_prop.maxTexture2D[1] << ")" << std::endl;
    std::cout << "Max Texture Dimension 3D: (" << device_prop.maxTexture3D[0] << ", "
              << device_prop.maxTexture3D[1] << ", "
              << device_prop.maxTexture3D[2] << ")" << std::endl;
    std::cout << "Smallest signal size: 128, largest signal size: 32768" << std::endl;
    std::vector<std::pair<int, int>> test_cases = {
        {128, 3}, {128, 5}, {128, 8}, {512, 5}, {512, 8}, {512, 11}, {1024, 11}, {1024, 15}, {1024, 20}, {1024, 31}, {2048, 31}, {2048, 44}, {2048, 63}, {4096, 63}, {4096, 127}, {8192, 127}, {8192, 255}, {16384, 255}, {16384, 511}, {32768, 511}, {32768, 1023}
    };

    for (const auto& [signal_size, filter_size] : test_cases) {
        benchmark_fft_convolution(signal_size, filter_size, 100, no_gpu, no_cpu);
    }
    // Additional test cases for larger sizes
    std::cout << "Additional test cases for larger sizes..." << std::endl;
    std::vector<std::pair<int, int>> additional_cases = {
        {65536, 1023}, {65536, 2047}, {131072, 2047}, {131072, 4095}, {262144, 4095}, {262144, 8191}, {524288, 8191}, {524288, 16383}, {1048576, 16383}, {1048576, 32767}, {2097152, 32767}, {2097152, 65535}
    };
    for (const auto& [signal_size, filter_size] : additional_cases) {
        benchmark_fft_convolution(signal_size, filter_size, 100, no_gpu, no_cpu);
    }
    // Very large test cases
    std::cout << "Very large test cases..." << std::endl;
    std::vector<std::pair<int, int>> very_large_cases = {
        {41943040, 65535}, {41943040, 131071}, {83886080, 131071}, {83886080, 262143}, {167772160, 262143}, {167772160, 524287}, {335544320, 524287}, {335544320, 1048575}
    };
    for (const auto& [signal_size, filter_size] : very_large_cases) {
        benchmark_fft_convolution(signal_size, filter_size, 100, no_gpu, no_cpu);
    }

    std::cout << "All benchmarks completed successfully." << std::endl;

    return 0;
}
