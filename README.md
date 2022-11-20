# BinaryMatcherCUDA
Binary descriptor matcher implemented using CUDA

This projects implements a simple binary descriptor matching (Hamming Distance) using CUDA.
Currently, implementation only supports Brute-Force Matching (Nearest Neighbour), but Nearest Neighbour matching could
be implemented easily.

I have implemented four CUDA kernels. One naive kernel as the base implementation, one where I make use of
the shared memory, one where I read in descriptors by 64-bit values and one where the global reads are optimized.

Results are compared with OpenCV CPU version and execution times are compared with OpenCV CUDA version.

Here are the current benchmark results on Nvidia 3060Ti, with 160.000 query and train descriptors:

Time to run naive kernel : 710.9888916 ms

Time to run shared memory kernel : 531.4229126 ms

Time to run shared memory 64 Bit kernel : 424.2384338 ms

Time to run shared memory 64 Bit Optimized kernel : 381.7770386 ms

openCVMatcher ran in  2487.15394 ms
