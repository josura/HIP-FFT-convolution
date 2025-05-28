# Project: HIP-based FFT Gated Convolution (FlashFFTConv-style)
# Description: High-performance HIP implementation of FFT-based gated convolution for long-context models

cmake_minimum_required(VERSION 3.21) # HIP language support requires 3.21
cmake_policy(VERSION 3.21.3...3.27)
project(HIPFFTConv LANGUAGES C CXX HIP)
# set the compiler to HIP or amdclang
set(CMAKE_CXX_COMPILER hipcc)
# adding nostartfiles to avoid linking issues with HIP runtime
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -nostartfiles")

# Enable HIP support
find_package(HIP REQUIRED)

# Set HIP compilation flags
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
# select backend for HIP, nvidia for NVIDIA GPUs, something else for AMD GPUs
# TODO

# Optional: Set HIP compiler explicitly (can help debugging or docker builds)
# set(CMAKE_CXX_COMPILER hipcc)  # Usually not needed if HIP is properly set up

# Determine backend platform
if(HIP_PLATFORM STREQUAL "amd")
    message(STATUS "Detected HIP Platform: AMD (ROCm)")
    set(HIP_BACKEND "rocm")
elseif(HIP_PLATFORM STREQUAL "nvidia")
    message(STATUS "Detected HIP Platform: NVIDIA (CUDA backend)")
    set(HIP_BACKEND "cuda")
else()
    message(WARNING "Unknown HIP platform, assuming default behavior.")
endif()
# possibly set the HIP_ARCHITECTURES variable to target specific GPU architectures

# Set build flags
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_HIP_STANDARD 17)

# Include subdirectories
add_subdirectory(src)
add_subdirectory(test)
add_subdirectory(examples)
include_directories(include)