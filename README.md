# HIP-FFT-convolution
Convolutions for Long Sequences, implemented in HIP, optimised for multi-gpu support

# BUILDING
To build the project, run the following command in the root directory of the project:
```bash
cmake -B build
cd build
make 
```

# RUNNING
To run the example, execute the following command in the `build` directory:
```bash
./build/examples/example_conv
```