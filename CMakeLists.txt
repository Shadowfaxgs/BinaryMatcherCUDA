cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

PROJECT(BinaryMatcherCUDA CXX CUDA) 

SET(CMAKE_CXX_COMPILER /usr/bin/gcc-8)
set(CMAKE_CXX_STANDARD 14)
SET(CMAKE_BUILD_TYPE Release)
SET(CMAKE_CUDA_FLAGS="-lineinfo -arch=sm_61 --ptxas-options=-v --use_fast_math")

find_package(OpenCV REQUIRED PATHS /home/ozgun/workspace/software/opencv_4.1.2_install)

find_package(CUDA)

file(GLOB source
    "source/*.cu"
    "source/*.cpp"
)

add_executable(BinaryMatcherCUDA ${source})

target_include_directories(BinaryMatcherCUDA PUBLIC include/ /home/ozgun/workspace/software/opencv_4.1.2_install/include /usr/local/cuda/samples/common/inc /usr/local/cuda/include)
target_compile_options(BinaryMatcherCUDA PRIVATE -std=c++14)

set_property(TARGET BinaryMatcherCUDA 
             PROPERTY CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(BinaryMatcherCUDA PRIVATE -lstdc++ cudart curand ${OpenCV_LIBS})

# We need to add the path to the driver (libcuda.dylib) as an rpath, 
#   so that the static cuda runtime can find it at runtime.
set_property(TARGET BinaryMatcherCUDA PROPERTY BUILD_RPATH ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})