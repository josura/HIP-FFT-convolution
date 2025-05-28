# CMakeLists.txt for hipfftconv library of kernels 
add_library(hipfftconv STATIC
    fft_utils.hip 
    conv_layer.hip
    fft_convolution.hip
)

find_library(HIPFFT_LIBRARY hipfft HINTS /opt/rocm/lib REQUIRED)

if(NOT HIPFFT_LIBRARY)
  message(FATAL_ERROR "hipfft library not found")
else()
  message(STATUS "Found hipfft library: ${HIPFFT_LIBRARY}")
endif()

target_link_libraries(hipfftconv PUBLIC ${HIPFFT_LIBRARY})


# target_include_directories(hipfftconv PUBLIC
#     #$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
#     #$<INSTALL_INTERFACE:include>
# )
target_include_directories(hipfftconv PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)

find_library(HIPFFT_LIBRARY hipfft HINTS /opt/rocm/lib /opt/rocm/lib64 REQUIRED)

target_link_libraries(hipfftconv PUBLIC
    hip::host
    hip::device
)