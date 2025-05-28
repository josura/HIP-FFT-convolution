# TODO add FFT convolution kernel and the convolutional layer kernel 
add_library(hipfftconv STATIC
    fft_utils.hip 
    conv_layer.hip
)
target_include_directories(hipfftconv PUBLIC
    #$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    #$<INSTALL_INTERFACE:include>
)
target_link_libraries(hipfftconv PUBLIC
    hip::host
    hip::device
)