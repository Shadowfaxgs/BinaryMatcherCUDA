#include <stdint.h>
#include <limits>
#include <chrono>
#include <iostream>
#include <iomanip>

#include <opencv2/opencv.hpp>
#include <cstring>
#include <string>
#include <vector>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define BLOCK_SIZE 32

__global__ void BinaryMatcherNaive(  const __restrict__ uint64_t* frameDescriptors, const __restrict__ uint64_t* databaseDescriptors, const size_t numberOfFrameDescriptors, 
                                const size_t numberOfDatabaseDescriptors, __restrict__ uint32_t* matches, __restrict__ uint16_t* distances)
{
    int threadId = blockDim.x * blockIdx.x + threadIdx.x;

    if(threadId < numberOfFrameDescriptors)
    {
        uint16_t distance;
        uint16_t minDistance = 513;
        uint32_t matchId = 0;

        for(size_t i = 0; i < numberOfDatabaseDescriptors; i++)
        {
            distance = 0;

            distance =  __popcll(frameDescriptors[threadId * 8]     ^ databaseDescriptors[i * 8]    );
            distance += __popcll(frameDescriptors[threadId * 8 + 1] ^ databaseDescriptors[i * 8 + 1]);
            distance += __popcll(frameDescriptors[threadId * 8 + 2] ^ databaseDescriptors[i * 8 + 2]);
            distance += __popcll(frameDescriptors[threadId * 8 + 3] ^ databaseDescriptors[i * 8 + 3]);
            distance += __popcll(frameDescriptors[threadId * 8 + 4] ^ databaseDescriptors[i * 8 + 4]);
            distance += __popcll(frameDescriptors[threadId * 8 + 5] ^ databaseDescriptors[i * 8 + 5]);
            distance += __popcll(frameDescriptors[threadId * 8 + 6] ^ databaseDescriptors[i * 8 + 6]);
            distance += __popcll(frameDescriptors[threadId * 8 + 7] ^ databaseDescriptors[i * 8 + 7]);
        
            if(distance < minDistance)
            {
                minDistance = distance;
                matchId = i;
            }
        }

        matches[threadId] = matchId;
        distances[threadId] = minDistance;
    }
}

__global__ void BinaryMatcherWithSharedMem(  const __restrict__ uint32_t *frameDescriptors, const __restrict__ uint32_t *databaseDescriptors, const size_t numberOfFrameDescriptors, 
                                const size_t numberOfDatabaseDescriptors, __restrict__ uint32_t *matches, __restrict__ uint16_t* distances)
{
    __shared__ uint32_t databaseDescriptorsShared[16 * BLOCK_SIZE];
    
    int threadId = blockDim.x * blockIdx.x + threadIdx.x;

    if(threadId < numberOfFrameDescriptors)
    {
        uint16_t distance;
        uint16_t minDistance = 513;
        uint32_t matchId = 0;

        for(size_t i = 0; i < numberOfDatabaseDescriptors; i += BLOCK_SIZE)
        {
            databaseDescriptorsShared[threadIdx.x * 16]      = databaseDescriptors[(i + threadIdx.x) * 16];
            databaseDescriptorsShared[threadIdx.x * 16 + 1 ] = databaseDescriptors[(i + threadIdx.x) * 16 + 1 ];
            databaseDescriptorsShared[threadIdx.x * 16 + 2 ] = databaseDescriptors[(i + threadIdx.x) * 16 + 2 ];
            databaseDescriptorsShared[threadIdx.x * 16 + 3 ] = databaseDescriptors[(i + threadIdx.x) * 16 + 3 ];
            databaseDescriptorsShared[threadIdx.x * 16 + 4 ] = databaseDescriptors[(i + threadIdx.x) * 16 + 4 ];
            databaseDescriptorsShared[threadIdx.x * 16 + 5 ] = databaseDescriptors[(i + threadIdx.x) * 16 + 5 ];
            databaseDescriptorsShared[threadIdx.x * 16 + 6 ] = databaseDescriptors[(i + threadIdx.x) * 16 + 6 ];
            databaseDescriptorsShared[threadIdx.x * 16 + 7 ] = databaseDescriptors[(i + threadIdx.x) * 16 + 7 ];
            databaseDescriptorsShared[threadIdx.x * 16 + 8 ] = databaseDescriptors[(i + threadIdx.x) * 16 + 8 ];
            databaseDescriptorsShared[threadIdx.x * 16 + 9 ] = databaseDescriptors[(i + threadIdx.x) * 16 + 9 ];
            databaseDescriptorsShared[threadIdx.x * 16 + 10] = databaseDescriptors[(i + threadIdx.x) * 16 + 10];
            databaseDescriptorsShared[threadIdx.x * 16 + 11] = databaseDescriptors[(i + threadIdx.x) * 16 + 11];
            databaseDescriptorsShared[threadIdx.x * 16 + 12] = databaseDescriptors[(i + threadIdx.x) * 16 + 12];
            databaseDescriptorsShared[threadIdx.x * 16 + 13] = databaseDescriptors[(i + threadIdx.x) * 16 + 13];
            databaseDescriptorsShared[threadIdx.x * 16 + 14] = databaseDescriptors[(i + threadIdx.x) * 16 + 14];
            databaseDescriptorsShared[threadIdx.x * 16 + 15] = databaseDescriptors[(i + threadIdx.x) * 16 + 15];

            __syncthreads();

            for(size_t j = 0; j < BLOCK_SIZE; j++)
            {
                distance = 0;

                distance =  __popc(frameDescriptors[threadId * 16]      ^ databaseDescriptorsShared[j * 16]);
                distance += __popc(frameDescriptors[threadId * 16 + 1 ] ^ databaseDescriptorsShared[j * 16 + 1 ]);
                distance += __popc(frameDescriptors[threadId * 16 + 2 ] ^ databaseDescriptorsShared[j * 16 + 2 ]);
                distance += __popc(frameDescriptors[threadId * 16 + 3 ] ^ databaseDescriptorsShared[j * 16 + 3 ]);
                distance += __popc(frameDescriptors[threadId * 16 + 4 ] ^ databaseDescriptorsShared[j * 16 + 4 ]);
                distance += __popc(frameDescriptors[threadId * 16 + 5 ] ^ databaseDescriptorsShared[j * 16 + 5 ]);
                distance += __popc(frameDescriptors[threadId * 16 + 6 ] ^ databaseDescriptorsShared[j * 16 + 6 ]);
                distance += __popc(frameDescriptors[threadId * 16 + 7 ] ^ databaseDescriptorsShared[j * 16 + 7 ]);
                distance += __popc(frameDescriptors[threadId * 16 + 8 ] ^ databaseDescriptorsShared[j * 16 + 8 ]);
                distance += __popc(frameDescriptors[threadId * 16 + 9 ] ^ databaseDescriptorsShared[j * 16 + 9 ]);
                distance += __popc(frameDescriptors[threadId * 16 + 10] ^ databaseDescriptorsShared[j * 16 + 10]);
                distance += __popc(frameDescriptors[threadId * 16 + 11] ^ databaseDescriptorsShared[j * 16 + 11]);
                distance += __popc(frameDescriptors[threadId * 16 + 12] ^ databaseDescriptorsShared[j * 16 + 12]);
                distance += __popc(frameDescriptors[threadId * 16 + 13] ^ databaseDescriptorsShared[j * 16 + 13]);
                distance += __popc(frameDescriptors[threadId * 16 + 14] ^ databaseDescriptorsShared[j * 16 + 14]);
                distance += __popc(frameDescriptors[threadId * 16 + 15] ^ databaseDescriptorsShared[j * 16 + 15]);

                if(distance < minDistance)
                {
                    matchId = i + j;
                    minDistance = distance;
                }
            }
        }

        matches[threadId] = matchId;
        distances[threadId] = minDistance;
    }
}

__global__ void BinaryMatcherWithSharedMem64Bit(  const __restrict__ uint64_t *frameDescriptors, const __restrict__ uint64_t *databaseDescriptors, const size_t numberOfFrameDescriptors, 
                                const size_t numberOfDatabaseDescriptors, __restrict__ uint32_t *matches, __restrict__ uint16_t* distances)
{
    __shared__ uint64_t databaseDescriptorsShared[8 * BLOCK_SIZE];
    
    int threadId = blockDim.x * blockIdx.x + threadIdx.x;

    if(threadId < numberOfFrameDescriptors)
    {
        uint16_t distance;
        uint16_t minDistance = 513;
        uint32_t matchId = 0;

        for(size_t i = 0; i < numberOfDatabaseDescriptors; i += BLOCK_SIZE)
        {
            databaseDescriptorsShared[threadIdx.x * 8]      = databaseDescriptors[(i + threadIdx.x) * 8];
            databaseDescriptorsShared[threadIdx.x * 8 + 1 ] = databaseDescriptors[(i + threadIdx.x) * 8 + 1 ];
            databaseDescriptorsShared[threadIdx.x * 8 + 2 ] = databaseDescriptors[(i + threadIdx.x) * 8 + 2 ];
            databaseDescriptorsShared[threadIdx.x * 8 + 3 ] = databaseDescriptors[(i + threadIdx.x) * 8 + 3 ];
            databaseDescriptorsShared[threadIdx.x * 8 + 4 ] = databaseDescriptors[(i + threadIdx.x) * 8 + 4 ];
            databaseDescriptorsShared[threadIdx.x * 8 + 5 ] = databaseDescriptors[(i + threadIdx.x) * 8 + 5 ];
            databaseDescriptorsShared[threadIdx.x * 8 + 6 ] = databaseDescriptors[(i + threadIdx.x) * 8 + 6 ];
            databaseDescriptorsShared[threadIdx.x * 8 + 7 ] = databaseDescriptors[(i + threadIdx.x) * 8 + 7 ];

            __syncthreads();

            for(size_t j = 0; j < BLOCK_SIZE; j++)
            {
                distance = 0;

                distance =  __popcll(frameDescriptors[threadId * 8]      ^ databaseDescriptorsShared[j * 8]);
                distance += __popcll(frameDescriptors[threadId * 8 + 1 ] ^ databaseDescriptorsShared[j * 8 + 1 ]);
                distance += __popcll(frameDescriptors[threadId * 8 + 2 ] ^ databaseDescriptorsShared[j * 8 + 2 ]);
                distance += __popcll(frameDescriptors[threadId * 8 + 3 ] ^ databaseDescriptorsShared[j * 8 + 3 ]);
                distance += __popcll(frameDescriptors[threadId * 8 + 4 ] ^ databaseDescriptorsShared[j * 8 + 4 ]);
                distance += __popcll(frameDescriptors[threadId * 8 + 5 ] ^ databaseDescriptorsShared[j * 8 + 5 ]);
                distance += __popcll(frameDescriptors[threadId * 8 + 6 ] ^ databaseDescriptorsShared[j * 8 + 6 ]);
                distance += __popcll(frameDescriptors[threadId * 8 + 7 ] ^ databaseDescriptorsShared[j * 8 + 7 ]);

                if(distance < minDistance)
                {
                    matchId = i + j;
                    minDistance = distance;
                }
            }
        }

        matches[threadId] = matchId;
        distances[threadId] = minDistance;
    }
}

int main()
{
    const size_t countOfFrameDescriptors = BLOCK_SIZE * 50;
    const size_t countOfDatabaseDescriptors = BLOCK_SIZE * 500;
    const size_t sizeOfOneDescriptor = 8 * sizeof(uint64_t);
    const size_t numberOfRuns = 100;

    uint64_t* frameDescriptors = new uint64_t[countOfFrameDescriptors * sizeOfOneDescriptor];
    uint64_t* databaseDescriptors = new uint64_t[countOfDatabaseDescriptors * sizeOfOneDescriptor];
    
    uint32_t* matchesNaive = new uint32_t[countOfFrameDescriptors];
    uint32_t* matchesSharedMem = new uint32_t[countOfFrameDescriptors];
    uint32_t* matchesSharedMem64Bit = new uint32_t[countOfFrameDescriptors];

    uint16_t* distancesNaive = new uint16_t[countOfFrameDescriptors];
    uint16_t* distancesSharedMem = new uint16_t[countOfFrameDescriptors];
    uint16_t* distancesSharedMem64Bit = new uint16_t[countOfFrameDescriptors];

    // Fill in descriptors
    srand(36);
    for (int i = 0; i < sizeOfOneDescriptor * countOfFrameDescriptors; ++i) 
    {
		reinterpret_cast<uint8_t*>(frameDescriptors)[i] = static_cast<uint8_t>(rand());
    }
    for (int i = 0; i < sizeOfOneDescriptor * countOfDatabaseDescriptors; ++i) 
    {
		reinterpret_cast<uint8_t*>(databaseDescriptors)[i] = static_cast<uint8_t>(rand());
    }
    
    uint64_t* deviceFrameDescriptors;
    uint64_t* deviceDatabaseDescriptors;
    uint32_t* deviceMatches;
    uint16_t* deviceDistances;
    cudaMalloc(&deviceFrameDescriptors, countOfFrameDescriptors * sizeOfOneDescriptor);
    cudaMalloc(&deviceDatabaseDescriptors, countOfDatabaseDescriptors * sizeOfOneDescriptor);
    cudaMalloc(&deviceMatches, countOfFrameDescriptors * sizeof(uint32_t));
    cudaMalloc(&deviceDistances, countOfFrameDescriptors * sizeof(uint16_t));

    cudaMemcpy(deviceFrameDescriptors, frameDescriptors, countOfFrameDescriptors * sizeOfOneDescriptor, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceDatabaseDescriptors, databaseDescriptors, countOfDatabaseDescriptors * sizeOfOneDescriptor, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds = 0;

    cudaEventRecord(start);

    // Run the naive kernel
    for(size_t i = 0; i < numberOfRuns; i++)
    {
        BinaryMatcherNaive<<<(countOfFrameDescriptors/BLOCK_SIZE) + 1, BLOCK_SIZE>>> (deviceFrameDescriptors, deviceDatabaseDescriptors, countOfFrameDescriptors, countOfDatabaseDescriptors, deviceMatches, deviceDistances);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);

    std::cout << "Time to run naive kernel : " << std::setprecision(10) << milliseconds / numberOfRuns << " ms" << std::endl;

    cudaMemcpy(matchesNaive, deviceMatches, countOfFrameDescriptors * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(distancesNaive, deviceDistances, countOfFrameDescriptors * sizeof(uint16_t), cudaMemcpyDeviceToHost);

    // Run the kernel with shared memory usage
    cudaMemset(deviceMatches, 0, countOfFrameDescriptors * sizeof(uint32_t));
    cudaMemset(deviceDistances, 0, countOfFrameDescriptors * sizeof(uint16_t));

    // Warm-up
    for(size_t i = 0; i < numberOfRuns; i++)
    {
        BinaryMatcherWithSharedMem<<<(countOfFrameDescriptors/BLOCK_SIZE) + 1, BLOCK_SIZE>>> (reinterpret_cast<uint32_t*>(deviceFrameDescriptors), reinterpret_cast<uint32_t*>(deviceDatabaseDescriptors), countOfFrameDescriptors, countOfDatabaseDescriptors, deviceMatches, deviceDistances);
        cudaDeviceSynchronize();
    }

    cudaEventRecord(start);

    for(size_t i = 0; i < numberOfRuns; i++)
    {
        BinaryMatcherWithSharedMem<<<(countOfFrameDescriptors/BLOCK_SIZE) + 1, BLOCK_SIZE>>> (reinterpret_cast<uint32_t*>(deviceFrameDescriptors), reinterpret_cast<uint32_t*>(deviceDatabaseDescriptors), countOfFrameDescriptors, countOfDatabaseDescriptors, deviceMatches, deviceDistances);
        cudaDeviceSynchronize();
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    std::cout << "Time to run shared memory kernel : " << std::setprecision(10) << milliseconds / numberOfRuns << " ms" << std::endl;

    cudaMemcpy(matchesSharedMem, deviceMatches, countOfFrameDescriptors * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(distancesSharedMem, deviceDistances, countOfFrameDescriptors * sizeof(uint16_t), cudaMemcpyDeviceToHost);

    // Run the kernel with shared memory usage and 64-bit computation
    cudaMemset(deviceMatches, 0, countOfFrameDescriptors * sizeof(uint32_t));
    cudaMemset(deviceDistances, 0, countOfFrameDescriptors * sizeof(uint16_t));

    // Warm-up
    for(size_t i = 0; i < numberOfRuns; i++)
    {
        BinaryMatcherWithSharedMem64Bit<<<(countOfFrameDescriptors/BLOCK_SIZE) + 1, BLOCK_SIZE>>> (deviceFrameDescriptors, deviceDatabaseDescriptors, countOfFrameDescriptors, countOfDatabaseDescriptors, deviceMatches, deviceDistances);
        cudaDeviceSynchronize();
    }

    cudaEventRecord(start);

    for(size_t i = 0; i < numberOfRuns; i++)
    {
        BinaryMatcherWithSharedMem64Bit<<<(countOfFrameDescriptors/BLOCK_SIZE) + 1, BLOCK_SIZE>>> (deviceFrameDescriptors, deviceDatabaseDescriptors, countOfFrameDescriptors, countOfDatabaseDescriptors, deviceMatches, deviceDistances);
        cudaDeviceSynchronize();
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    std::cout << "Time to run shared memory 64 Bit kernel : " << std::setprecision(10) << milliseconds / numberOfRuns << " ms" << std::endl;

    cudaMemcpy(matchesSharedMem64Bit, deviceMatches, countOfFrameDescriptors * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(distancesSharedMem64Bit, deviceDistances, countOfFrameDescriptors * sizeof(uint16_t), cudaMemcpyDeviceToHost);

    // Run OpenCV based matcher
    cv::Ptr<cv::cuda::DescriptorMatcher> openCVGPUMatcher = cv::cuda::DescriptorMatcher::createBFMatcher(cv::NORM_HAMMING);
    cv::Ptr<cv::DescriptorMatcher> openCVMatcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::MatcherType::BRUTEFORCE_HAMMING);

    cv::cuda::GpuMat frameGPUMat(countOfFrameDescriptors, sizeOfOneDescriptor, CV_8U, deviceFrameDescriptors);
    cv::cuda::GpuMat databaseGPUMat(countOfDatabaseDescriptors, sizeOfOneDescriptor, CV_8U, deviceDatabaseDescriptors);
    cv::Mat frameMat(countOfFrameDescriptors, sizeOfOneDescriptor, CV_8U, frameDescriptors);
    cv::Mat databaseMat(countOfDatabaseDescriptors, sizeOfOneDescriptor, CV_8U, databaseDescriptors);
    std::vector<cv::DMatch> openCVGPUMatches;
    std::vector<cv::DMatch> openCVCPUMatches;

    // Warm-up
    for(size_t i = 0; i < numberOfRuns; i++)
    {
        openCVGPUMatcher->match(frameGPUMat, databaseGPUMat, openCVGPUMatches);
    }

    std::chrono::high_resolution_clock::time_point startTime = std::chrono::high_resolution_clock::now();

    for(size_t i = 0; i < numberOfRuns; i++)
    {
        openCVGPUMatcher->match(frameGPUMat, databaseGPUMat, openCVGPUMatches);
    }

    std::chrono::high_resolution_clock::time_point endTime = std::chrono::high_resolution_clock::now();
    double sec = static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(endTime - startTime).count()) * 1e-9 / static_cast<double>(numberOfRuns);
	std::cout << "openCVMatcher ran in  "  << sec * 1e3 << " ms" << std::endl;

    for(size_t i = 0; i < numberOfRuns; i++)
    {
        openCVMatcher->match(frameMat, databaseMat, openCVCPUMatches);
    }

    // Compare results with ground-truth (OpenCV)
    for(int i = 0; i < countOfFrameDescriptors; i++)
    {
        if(matchesNaive[i] != openCVCPUMatches[i].trainIdx)
        {
            std::cout   << "queryIdx = " << i
                        << " matchesNaive[i] := " << matchesNaive[i] 
                        << " distance := " << distancesNaive[i] 
                        << " and openCVCPUMatches[i].trainIdx = " << openCVCPUMatches[i].trainIdx 
                        << " distance = " << openCVCPUMatches[i].distance
                        << " do not match!" << std::endl;
        }
        
        if(matchesSharedMem[i] != openCVCPUMatches[i].trainIdx)
        {
            std::cout   << "queryIdx = " << i
                        << " matchesSharedMem[i] := " << matchesSharedMem[i] 
                        << " distance := " << distancesSharedMem[i] 
                        << " and openCVCPUMatches[i].trainIdx = " << openCVCPUMatches[i].trainIdx 
                        << " distance = " << openCVCPUMatches[i].distance
                        << " do not match!" << std::endl;
        }

        if(matchesSharedMem64Bit[i] != openCVCPUMatches[i].trainIdx)
        {
            std::cout   << "queryIdx = " << i
                        << " matchesSharedMem64Bit[i] := " << matchesSharedMem64Bit[i] 
                        << " distance := " << distancesSharedMem64Bit[i] 
                        << " and openCVCPUMatches[i].trainIdx = " << openCVCPUMatches[i].trainIdx 
                        << " distance = " << openCVCPUMatches[i].distance
                        << " do not match!" << std::endl;
        }
    }

    return 0;
}