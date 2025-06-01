# HIP-FFT-convolution
Convolutions for Long Sequences, implemented in HIP, optimised for multi-gpu support. Under development 

# BUILDING
To build the project, run the following command in the root directory of the project:
```bash
cmake -B build
cd build
make 
```

To build the project with CUDA support, you need hipfft installed, and change the library and includes inside the CMakeLists.txt and src/CMakeLists.txt you can use the following command:
```bash
cmake -B build -DHIP_PLATFORM=nvidia -DHIP_COMPILER=nvcc
cd build
make
```

# RUNNING
To run the example, execute the following command in the `build` directory:
```bash
./build/examples/example_conv
```

# BENCHMARKING
To run the benchmark, execute the following command in the `build` directory:
```bash
./build/benchmark/benchmark_fft
```
To see options for the benchmark, run:
```bash
./build/benchmark/benchmark_fft --help
```