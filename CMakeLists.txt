# Project: HIP-based FFT Gated Convolution (FlashFFTConv-style)
# Description: High-performance HIP implementation of FFT-based gated convolution for long-context models

cmake_minimum_required(VERSION 3.21) # HIP language support requires 3.21
cmake_policy(VERSION 3.21.3...3.27)
project(HIPFFTConv LANGUAGES C CXX HIP)
# set the compiler to HIP or amdclang
set(CMAKE_CXX_COMPILER hipcc)

# Enable HIP support
find_package(HIP REQUIRED)

# Set HIP compilation flags
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
# possibly set the HIP_ARCHITECTURES variable to target specific GPU architectures

# Set build flags
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_HIP_STANDARD 17)

# Include subdirectories
add_subdirectory(src)
add_subdirectory(test)
add_subdirectory(examples)