# Project: HIP-based FFT Gated Convolution (FlashFFTConv-style)
# Description: High-performance HIP implementation of FFT-based gated convolution for long-context models

cmake_minimum_required(VERSION 3.20)
project(HIPFFTConv LANGUAGES C CXX HIP)

# Enable HIP support
find_package(HIP REQUIRED)

# Set HIP compilation flags
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
# possibly set the HIP_ARCHITECTURES variable to target specific GPU architectures