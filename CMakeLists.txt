# Project: HIP-based FFT Gated Convolution (FlashFFTConv-style)
# Description: High-performance HIP implementation of FFT-based gated convolution for long-context models

cmake_minimum_required(VERSION 3.21) # HIP language support requires 3.21
cmake_policy(VERSION 3.21.3...3.27)
project(HIPFFTConv LANGUAGES C CXX HIP)
# set the compiler to HIP or amdclang
set(CMAKE_CXX_COMPILER hipcc)
# adding nostartfiles to avoid linking issues with HIP runtime
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -nostartfiles")

# Enable HIP support
find_package(HIP REQUIRED)

# Set HIP compilation flags
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_HIP_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)  # disables gnu++XX and forces c++XX
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

# Detect platform
if(HIP_PLATFORM STREQUAL "nvidia")
    message(STATUS "Using HIP platform: NVIDIA")
    # setting position independent code off for NVIDIA (not supported by nvcc)
    set(CMAKE_POSITION_INDEPENDENT_CODE OFF)
    
    # Remove flags that nvcc doesn't support
    foreach(flag_var
        CMAKE_CXX_FLAGS
        CMAKE_CXX_FLAGS_DEBUG
        CMAKE_CXX_FLAGS_RELEASE
        CMAKE_EXE_LINKER_FLAGS
    )
        if(DEFINED ${flag_var})
            string(REPLACE "-fPIE" "" ${flag_var} "${${flag_var}}")
            string(REPLACE "-nostartfiles" "" ${flag_var} "${${flag_var}}")
        endif()
    endforeach()
    # Check for nvcc problem with math.h
    # Append system include path for nvcc to avoid math.h error
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -isystem /usr/include")
    # This is a workaround for nvcc not finding math.h in some environments
    include(CheckIncludeFile)
    check_include_file("math.h" HAS_MATH_H)
    if(NOT HAS_MATH_H)
        include_directories("/usr/include") # might be "/usr/include/x86_64-linux-gnu" on Debian
    endif()
    include_directories(BEFORE "/home/josura/opt/hipFFT/build/include")
    link_directories("/home/josura/opt/hipFFT/build/library")
    message(STATUS "HIPFFT include path: /home/josura/opt/hipFFT/build/include")
    message(STATUS "HIPFFT library path: /home/josura/opt/hipFFT/build/library")
endif()

# For debugging purposes, print out the compiler and flags
message(STATUS "CMAKE_CXX_FLAGS: ${CMAKE_CXX_FLAGS}")
message(STATUS "CMAKE_EXE_LINKER_FLAGS: ${CMAKE_EXE_LINKER_FLAGS}")


# possibly set the HIP_ARCHITECTURES variable to target specific GPU architectures

# Include subdirectories
add_subdirectory(src)
add_subdirectory(test)
add_subdirectory(examples)
add_subdirectory(benchmark)
include_directories(include)