/**
 * @file test_hip.cxx
 * @brief A simple test program to check if HIP is working correctly.
 * @details This program attempts to get the number of HIP devices available on the system.
 */
#include <hip/hip_runtime.h>
#include <iostream>

int main() {
    int deviceCount = 0;
    hipError_t err = hipGetDeviceCount(&deviceCount);
    if (err != hipSuccess) {
        std::cerr << "hipGetDeviceCount failed: " << hipGetErrorString(err) << "\n";
        return 1;
    }
    std::cout << "Number of HIP devices: " << deviceCount << "\n";
    return 0;
}