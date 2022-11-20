# BinaryMatcherCUDA
Binary descriptor matcher implemented using CUDA

This projects implements a simple binary descriptor matching (Hamming Distance) using CUDA.
Currently, implementation only supports Brute-Force Matching (Nearest Neighbour), but Nearest Neighbour matching could
be implemented easily.

I have implemented three CUDA kernels. One naive kernel as the base implementation, one where I make use of
the shared memory and one where I read in descriptors by 64-bit values.

Results are compared with OpenCV CPU version and execution times are compared with OpenCV CUDA version.

Here are the current benchmark results on Nvidia 3060Ti, with 160.000 query and train descriptors:

Time to run naive kernel : 708.4364014 ms

Time to run shared memory kernel : 551.4415283 ms

Time to run shared memory 64 Bit kernel : 441.7418518 ms

Time to run shared memory 64 Bit Transposition kernel : 373.2445984 ms

openCVMatcher ran in  2480.66902 ms
