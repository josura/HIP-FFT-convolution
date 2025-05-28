# CMakeLists.txt for hipfftconv library of kernels 
add_library(hipfftconv STATIC
    fft_utils.hip 
    conv_layer.hip
    fft_convolution.hip
)
# target_include_directories(hipfftconv PUBLIC
#     #$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
#     #$<INSTALL_INTERFACE:include>
# )
target_include_directories(hipfftconv PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)

target_link_libraries(hipfftconv PUBLIC
    hip::host
    hip::device
)