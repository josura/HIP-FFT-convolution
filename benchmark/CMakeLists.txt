# add_executable(benchmark_fft benchmark_fft.cxx)
# target_link_libraries(benchmark_fft hipfftconv hip::host hip::device)

add_executable(benchmark_fft benchmark_fft.cxx)

# Link to the hipfftconv library
target_link_libraries(benchmark_fft PRIVATE hipfftconv)
