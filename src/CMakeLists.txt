# # CMakeLists.txt for hipfftconv library of kernels 
# add_library(hipfftconv STATIC
#     fft_utils.hip 
#     conv_layer.hip
#     fft_convolution.hip
# )

# find_library(HINTS /opt/rocm/lib REQUIRED)
# target_link_libraries(hipfftconv PRIVATE hipfft)
# if(NOT HIPFFT_LIBRARY)
#   message(FATAL_ERROR "hipfft library not found")
# else()
#   message(STATUS "Found hipfft library: ${HIPFFT_LIBRARY}")
# endif()

# target_link_libraries(hipfftconv PUBLIC ${HIPFFT_LIBRARY})


# # target_include_directories(hipfftconv PUBLIC
# #     #$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
# #     #$<INSTALL_INTERFACE:include>
# # )
# target_include_directories(hipfftconv PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)

# find_library(HIPFFT_LIBRARY hipfft HINTS /opt/rocm/lib /opt/rocm/lib64 REQUIRED)

# target_link_libraries(hipfftconv PUBLIC
#     hip::host
#     hip::device
# )

# CMakeLists.txt for hipfftconv library of kernels 

# src/CMakeLists.txt - HIPFFTConv library for AMD and NVIDIA

add_library(hipfftconv STATIC
    fft_utils.hip 
    conv_layer.hip
    fft_convolution.hip
    naive_convolution_1d.hip
    naive_convolution_2d.hip
)

target_include_directories(hipfftconv PUBLIC ${CMAKE_SOURCE_DIR}/include)

# Detect HIP backend platform
if(HIP_PLATFORM STREQUAL "amd")
    message(STATUS "Configuring for AMD ROCm runtime")

    # Try to find system hipfft (usually installed with ROCm)
    find_library(HIPFFT_LIBRARY
        NAMES hipfft
        HINTS /opt/rocm/lib /opt/rocm/lib64
        REQUIRED
    )
    target_link_libraries(hipfftconv PRIVATE ${HIPFFT_LIBRARY})
    message(STATUS "Using ROCm hipfft: ${HIPFFT_LIBRARY}")

elseif(HIP_PLATFORM STREQUAL "nvidia")
    message(STATUS "Configuring for NVIDIA CUDA runtime")

    # Manually specify your local hipFFT paths
    set(HIPFFT_INCLUDE_DIR "/home/josura/opt/hipFFT/build/include")
    set(HIPFFT_LIBRARY_DIR "/home/josura/opt/hipFFT/build/library")
    set(HIPFFT_LIBRARY hipfft)

    target_include_directories(hipfftconv PUBLIC ${HIPFFT_INCLUDE_DIR})
    link_directories(${HIPFFT_LIBRARY_DIR})
    target_link_libraries(hipfftconv PRIVATE ${HIPFFT_LIBRARY})

    message(STATUS "Using custom-built hipfft for NVIDIA from ${HIPFFT_LIBRARY_DIR}")
else()
    message(FATAL_ERROR "Unknown HIP platform. Set HIP_PLATFORM to 'amd' or 'nvidia'")
endif()
