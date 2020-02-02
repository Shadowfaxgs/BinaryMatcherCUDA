# BinaryMatcherCUDA
Binary descriptor matcher implemented using CUDA

This projects implements a simple binary descriptor matching (Hamming Distance) using CUDA.
Currently, implementation only supports Brute-Force Matching (Nearest Neighbour), but Nearest Neighbour matching could
be implemented easily.

I have implemented three CUDA kernels. One naive kernel as the base implementation, one where I make use of
the shared memory and one where I read in descriptors by 64-bit values.

Results are compared with OpenCV CPU version and execution times are compared with OpenCV CUDA version.

Here are the current benchmark results:

Time to run naive kernel : 4.68009758 ms

Time to run shared memory kernel : 3.240493774 ms

Time to run shared memory 64 Bit kernel : 1.736196399 ms

openCVMatcher ran in  3.11371608 ms
