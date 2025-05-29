
enable_testing()
add_executable(test_fft test_fft.cxx)
add_executable(test_hip test_hip.cxx)
target_link_libraries(test_fft hipfftconv hip::host hip::device)
target_link_libraries(test_hip hip::host hip::device)
# TODO link some FFT library if needed, for validation purposes, e.g., hipfft or rocFFT
add_test(NAME test_fft COMMAND test_fft)
add_test(NAME test_hip COMMAND test_hip)