add_executable(example_conv example_1.cxx)
target_link_libraries(example_conv
    hipfftconv
    hip::host
    hip::device
)