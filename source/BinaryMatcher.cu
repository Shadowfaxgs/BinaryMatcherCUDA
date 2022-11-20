#include <stdint.h>
#include <limits>
#include <chrono>
#include <iostream>
#include <iomanip>

#include <cstring>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>
#include <opencv2/cudafeatures2d.hpp>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define BLOCK_SIZE 32

__global__ void BinaryMatcherNaive(  const uint64_t* __restrict__ queryDescriptors, const uint64_t* __restrict__ trainDescriptors, const size_t numberOfQueryDescriptors,
                                const size_t numberOfTrainDescriptors, uint32_t* __restrict__ matches, uint16_t* __restrict__ distances)
{
    int threadId = blockDim.x * blockIdx.x + threadIdx.x;

    if(threadId < numberOfQueryDescriptors)
    {
        uint16_t distance;
        uint16_t minDistance = 513;
        uint32_t matchId = 0;

        for(size_t i = 0; i < numberOfTrainDescriptors; i++)
        {
            distance = 0;

            distance =  __popcll(queryDescriptors[threadId * 8]     ^ trainDescriptors[i * 8]    );
            distance += __popcll(queryDescriptors[threadId * 8 + 1] ^ trainDescriptors[i * 8 + 1]);
            distance += __popcll(queryDescriptors[threadId * 8 + 2] ^ trainDescriptors[i * 8 + 2]);
            distance += __popcll(queryDescriptors[threadId * 8 + 3] ^ trainDescriptors[i * 8 + 3]);
            distance += __popcll(queryDescriptors[threadId * 8 + 4] ^ trainDescriptors[i * 8 + 4]);
            distance += __popcll(queryDescriptors[threadId * 8 + 5] ^ trainDescriptors[i * 8 + 5]);
            distance += __popcll(queryDescriptors[threadId * 8 + 6] ^ trainDescriptors[i * 8 + 6]);
            distance += __popcll(queryDescriptors[threadId * 8 + 7] ^ trainDescriptors[i * 8 + 7]);
        
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

__global__ void BinaryMatcherWithSharedMem(  const  uint32_t * __restrict__ queryDescriptors, const uint32_t * __restrict__ trainDescriptors, const size_t numberOfQueryDescriptors,
                                const size_t numberOfTrainDescriptors, uint32_t * __restrict__ matches,  uint16_t* __restrict__ distances)
{
    __shared__ uint32_t trainDescriptorsShared[16 * BLOCK_SIZE];
    
    int threadId = blockDim.x * blockIdx.x + threadIdx.x;

    if(threadId < numberOfQueryDescriptors)
    {
        uint16_t distance;
        uint16_t minDistance = 513;
        uint32_t matchId = 0;

        for(size_t i = 0; i < numberOfTrainDescriptors; i += BLOCK_SIZE)
        {
            trainDescriptorsShared[threadIdx.x * 16]      = trainDescriptors[(i + threadIdx.x) * 16];
            trainDescriptorsShared[threadIdx.x * 16 + 1 ] = trainDescriptors[(i + threadIdx.x) * 16 + 1 ];
            trainDescriptorsShared[threadIdx.x * 16 + 2 ] = trainDescriptors[(i + threadIdx.x) * 16 + 2 ];
            trainDescriptorsShared[threadIdx.x * 16 + 3 ] = trainDescriptors[(i + threadIdx.x) * 16 + 3 ];
            trainDescriptorsShared[threadIdx.x * 16 + 4 ] = trainDescriptors[(i + threadIdx.x) * 16 + 4 ];
            trainDescriptorsShared[threadIdx.x * 16 + 5 ] = trainDescriptors[(i + threadIdx.x) * 16 + 5 ];
            trainDescriptorsShared[threadIdx.x * 16 + 6 ] = trainDescriptors[(i + threadIdx.x) * 16 + 6 ];
            trainDescriptorsShared[threadIdx.x * 16 + 7 ] = trainDescriptors[(i + threadIdx.x) * 16 + 7 ];
            trainDescriptorsShared[threadIdx.x * 16 + 8 ] = trainDescriptors[(i + threadIdx.x) * 16 + 8 ];
            trainDescriptorsShared[threadIdx.x * 16 + 9 ] = trainDescriptors[(i + threadIdx.x) * 16 + 9 ];
            trainDescriptorsShared[threadIdx.x * 16 + 10] = trainDescriptors[(i + threadIdx.x) * 16 + 10];
            trainDescriptorsShared[threadIdx.x * 16 + 11] = trainDescriptors[(i + threadIdx.x) * 16 + 11];
            trainDescriptorsShared[threadIdx.x * 16 + 12] = trainDescriptors[(i + threadIdx.x) * 16 + 12];
            trainDescriptorsShared[threadIdx.x * 16 + 13] = trainDescriptors[(i + threadIdx.x) * 16 + 13];
            trainDescriptorsShared[threadIdx.x * 16 + 14] = trainDescriptors[(i + threadIdx.x) * 16 + 14];
            trainDescriptorsShared[threadIdx.x * 16 + 15] = trainDescriptors[(i + threadIdx.x) * 16 + 15];

            __syncthreads();

            for(size_t j = 0; j < BLOCK_SIZE; j++)
            {
                distance = 0;

                distance =  __popc(queryDescriptors[threadId * 16]      ^ trainDescriptorsShared[j * 16]);
                distance += __popc(queryDescriptors[threadId * 16 + 1 ] ^ trainDescriptorsShared[j * 16 + 1 ]);
                distance += __popc(queryDescriptors[threadId * 16 + 2 ] ^ trainDescriptorsShared[j * 16 + 2 ]);
                distance += __popc(queryDescriptors[threadId * 16 + 3 ] ^ trainDescriptorsShared[j * 16 + 3 ]);
                distance += __popc(queryDescriptors[threadId * 16 + 4 ] ^ trainDescriptorsShared[j * 16 + 4 ]);
                distance += __popc(queryDescriptors[threadId * 16 + 5 ] ^ trainDescriptorsShared[j * 16 + 5 ]);
                distance += __popc(queryDescriptors[threadId * 16 + 6 ] ^ trainDescriptorsShared[j * 16 + 6 ]);
                distance += __popc(queryDescriptors[threadId * 16 + 7 ] ^ trainDescriptorsShared[j * 16 + 7 ]);
                distance += __popc(queryDescriptors[threadId * 16 + 8 ] ^ trainDescriptorsShared[j * 16 + 8 ]);
                distance += __popc(queryDescriptors[threadId * 16 + 9 ] ^ trainDescriptorsShared[j * 16 + 9 ]);
                distance += __popc(queryDescriptors[threadId * 16 + 10] ^ trainDescriptorsShared[j * 16 + 10]);
                distance += __popc(queryDescriptors[threadId * 16 + 11] ^ trainDescriptorsShared[j * 16 + 11]);
                distance += __popc(queryDescriptors[threadId * 16 + 12] ^ trainDescriptorsShared[j * 16 + 12]);
                distance += __popc(queryDescriptors[threadId * 16 + 13] ^ trainDescriptorsShared[j * 16 + 13]);
                distance += __popc(queryDescriptors[threadId * 16 + 14] ^ trainDescriptorsShared[j * 16 + 14]);
                distance += __popc(queryDescriptors[threadId * 16 + 15] ^ trainDescriptorsShared[j * 16 + 15]);

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

__global__ void BinaryMatcherWithSharedMem64Bit(  const uint64_t* __restrict__ queryDescriptors, const uint64_t* __restrict__ trainDescriptors, const size_t numberOfQueryDescriptors,
                                const size_t numberOfTrainDescriptors, uint32_t* __restrict__ matches, uint16_t* __restrict__ distances)
{
    __shared__ uint64_t trainDescriptorsShared[8 * BLOCK_SIZE];
    
    int threadId = blockDim.x * blockIdx.x + threadIdx.x;

    if(threadId < numberOfQueryDescriptors)
    {
        uint16_t distance;
        uint16_t minDistance = 513;
        uint32_t matchId = 0;

        for(size_t i = 0; i < numberOfTrainDescriptors; i += BLOCK_SIZE)
        {
            trainDescriptorsShared[threadIdx.x * 8]      = trainDescriptors[(i + threadIdx.x) * 8];
            trainDescriptorsShared[threadIdx.x * 8 + 1 ] = trainDescriptors[(i + threadIdx.x) * 8 + 1 ];
            trainDescriptorsShared[threadIdx.x * 8 + 2 ] = trainDescriptors[(i + threadIdx.x) * 8 + 2 ];
            trainDescriptorsShared[threadIdx.x * 8 + 3 ] = trainDescriptors[(i + threadIdx.x) * 8 + 3 ];
            trainDescriptorsShared[threadIdx.x * 8 + 4 ] = trainDescriptors[(i + threadIdx.x) * 8 + 4 ];
            trainDescriptorsShared[threadIdx.x * 8 + 5 ] = trainDescriptors[(i + threadIdx.x) * 8 + 5 ];
            trainDescriptorsShared[threadIdx.x * 8 + 6 ] = trainDescriptors[(i + threadIdx.x) * 8 + 6 ];
            trainDescriptorsShared[threadIdx.x * 8 + 7 ] = trainDescriptors[(i + threadIdx.x) * 8 + 7 ];

            __syncthreads();

            for(size_t j = 0; j < BLOCK_SIZE; j++)
            {
                distance = 0;

                // if(threadId == 0)
                // {
                //     printf("SH j %lu \n", j);
                //     printf("SH 0 train part %lu \n", trainDescriptorsShared[j * 8]);
                //     printf("SH 1 train part %lu \n", trainDescriptorsShared[j * 8 + 1]);
                //     printf("SH 2 train part %lu \n", trainDescriptorsShared[j * 8 + 2]);
                //     printf("SH 3 train part %lu \n", trainDescriptorsShared[j * 8 + 3]);
                //     printf("SH 4 train part %lu \n", trainDescriptorsShared[j * 8 + 4]);
                //     printf("SH 5 train part %lu \n", trainDescriptorsShared[j * 8 + 5]);
                //     printf("SH 6 train part %lu \n", trainDescriptorsShared[j * 8 + 6]);
                //     printf("SH 7 train part %lu \n", trainDescriptorsShared[j * 8 + 7]);
                // }

                distance =  __popcll(queryDescriptors[threadId * 8]      ^ trainDescriptorsShared[j * 8]);
                distance += __popcll(queryDescriptors[threadId * 8 + 1 ] ^ trainDescriptorsShared[j * 8 + 1 ]);
                distance += __popcll(queryDescriptors[threadId * 8 + 2 ] ^ trainDescriptorsShared[j * 8 + 2 ]);
                distance += __popcll(queryDescriptors[threadId * 8 + 3 ] ^ trainDescriptorsShared[j * 8 + 3 ]);
                distance += __popcll(queryDescriptors[threadId * 8 + 4 ] ^ trainDescriptorsShared[j * 8 + 4 ]);
                distance += __popcll(queryDescriptors[threadId * 8 + 5 ] ^ trainDescriptorsShared[j * 8 + 5 ]);
                distance += __popcll(queryDescriptors[threadId * 8 + 6 ] ^ trainDescriptorsShared[j * 8 + 6 ]);
                distance += __popcll(queryDescriptors[threadId * 8 + 7 ] ^ trainDescriptorsShared[j * 8 + 7 ]);

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

__global__ void BinaryMatcherWithSharedMem64BitTranspose(  const uint64_t* __restrict__ queryDescriptors, const uint64_t* __restrict__ trainDescriptors, const size_t numberOfQueryDescriptors,
                                const size_t numberOfTrainDescriptors, uint32_t* __restrict__ matches, uint16_t* __restrict__ distances)
{   
    __shared__ uint64_t trainDescriptorsShared[8 * BLOCK_SIZE];
    
    int threadId = blockDim.x * blockIdx.x + threadIdx.x;

    if(threadId < numberOfQueryDescriptors)
    {
        uint16_t distance;
        uint16_t minDistance = 513;
        uint32_t matchId = 0;

        for(size_t i = 0; i < numberOfTrainDescriptors; i += BLOCK_SIZE)
        {
            __syncthreads();

            // printf("TR threadIdx.x %i \n", threadIdx.x);

            // if(threadId == 0)
            // {
            //     printf("TR index 0 %i \n", i + threadIdx.x + BLOCK_SIZE * 0);
            //     printf("TR index 1 %i \n", i + threadIdx.x + BLOCK_SIZE * 1);
            //     printf("TR index 2 %i \n", i + threadIdx.x + BLOCK_SIZE * 2);
            //     printf("TR index 3 %i \n", i + threadIdx.x + BLOCK_SIZE * 3);
            //     printf("TR index 4 %i \n", i + threadIdx.x + BLOCK_SIZE * 4);
            //     printf("TR index 5 %i \n", i + threadIdx.x + BLOCK_SIZE * 5);
            //     printf("TR index 6 %i \n", i + threadIdx.x + BLOCK_SIZE * 6);
            //     printf("TR index 7 %i \n", i + threadIdx.x + BLOCK_SIZE * 7); 
            // }
            trainDescriptorsShared[threadIdx.x + BLOCK_SIZE * 0] = trainDescriptors[i * 8 + threadIdx.x + BLOCK_SIZE * 0];
            trainDescriptorsShared[threadIdx.x + BLOCK_SIZE * 1] = trainDescriptors[i * 8 + threadIdx.x + BLOCK_SIZE * 1];
            trainDescriptorsShared[threadIdx.x + BLOCK_SIZE * 2] = trainDescriptors[i * 8 + threadIdx.x + BLOCK_SIZE * 2];
            trainDescriptorsShared[threadIdx.x + BLOCK_SIZE * 3] = trainDescriptors[i * 8 + threadIdx.x + BLOCK_SIZE * 3];
            trainDescriptorsShared[threadIdx.x + BLOCK_SIZE * 4] = trainDescriptors[i * 8 + threadIdx.x + BLOCK_SIZE * 4];
            trainDescriptorsShared[threadIdx.x + BLOCK_SIZE * 5] = trainDescriptors[i * 8 + threadIdx.x + BLOCK_SIZE * 5];
            trainDescriptorsShared[threadIdx.x + BLOCK_SIZE * 6] = trainDescriptors[i * 8 + threadIdx.x + BLOCK_SIZE * 6];
            trainDescriptorsShared[threadIdx.x + BLOCK_SIZE * 7] = trainDescriptors[i * 8 + threadIdx.x + BLOCK_SIZE * 7];

            __syncthreads();

            for(size_t j = 0; j < BLOCK_SIZE; j++)
            {
                distance = 0;

                // if(threadId == 0)
                // {
                //     printf("TR j %lu \n", j);
                //     printf("TR 0 train part %lu \n", trainDescriptorsShared[j * 8]);
                //     printf("TR 1 train part %lu \n", trainDescriptorsShared[j * 8 + 1]);
                //     printf("TR 2 train part %lu \n", trainDescriptorsShared[j * 8 + 2]);
                //     printf("TR 3 train part %lu \n", trainDescriptorsShared[j * 8 + 3]);
                //     printf("TR 4 train part %lu \n", trainDescriptorsShared[j * 8 + 4]);
                //     printf("TR 5 train part %lu \n", trainDescriptorsShared[j * 8 + 5]);
                //     printf("TR 6 train part %lu \n", trainDescriptorsShared[j * 8 + 6]);
                //     printf("TR 7 train part %lu \n", trainDescriptorsShared[j * 8 + 7]);
                // }
                
                distance =  __popcll(queryDescriptors[threadId * 8]      ^ trainDescriptorsShared[j * 8]);
                distance += __popcll(queryDescriptors[threadId * 8 + 1 ] ^ trainDescriptorsShared[j * 8 + 1 ]);
                distance += __popcll(queryDescriptors[threadId * 8 + 2 ] ^ trainDescriptorsShared[j * 8 + 2 ]);
                distance += __popcll(queryDescriptors[threadId * 8 + 3 ] ^ trainDescriptorsShared[j * 8 + 3 ]);
                distance += __popcll(queryDescriptors[threadId * 8 + 4 ] ^ trainDescriptorsShared[j * 8 + 4 ]);
                distance += __popcll(queryDescriptors[threadId * 8 + 5 ] ^ trainDescriptorsShared[j * 8 + 5 ]);
                distance += __popcll(queryDescriptors[threadId * 8 + 6 ] ^ trainDescriptorsShared[j * 8 + 6 ]);
                distance += __popcll(queryDescriptors[threadId * 8 + 7 ] ^ trainDescriptorsShared[j * 8 + 7 ]);

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
    const size_t countOfQueryDescriptors = BLOCK_SIZE * 5000;
    const size_t countOfTrainDescriptors = BLOCK_SIZE * 5000;
    const size_t descriptorSizeUint64 = 8;
    const size_t sizeOfOneDescriptor = descriptorSizeUint64 * sizeof(uint64_t);
    const size_t numberOfRuns = 10;

    uint64_t* queryDescriptors = new uint64_t[countOfQueryDescriptors * sizeOfOneDescriptor];
    uint64_t* trainDescriptors = new uint64_t[countOfTrainDescriptors * sizeOfOneDescriptor];
    
    uint32_t* matchesCPU = new uint32_t[countOfQueryDescriptors];
    uint32_t* matchesNaive = new uint32_t[countOfQueryDescriptors];
    uint32_t* matchesSharedMem = new uint32_t[countOfQueryDescriptors];
    uint32_t* matchesSharedMem64Bit = new uint32_t[countOfQueryDescriptors];
    uint32_t* matchesSharedMem64BitTranspose = new uint32_t[countOfQueryDescriptors];

    uint16_t* distancesCPU = new uint16_t[countOfQueryDescriptors];
    uint16_t* distancesNaive = new uint16_t[countOfQueryDescriptors];
    uint16_t* distancesSharedMem = new uint16_t[countOfQueryDescriptors];
    uint16_t* distancesSharedMem64Bit = new uint16_t[countOfQueryDescriptors];
    uint16_t* distancesSharedMem64BitTranspose = new uint16_t[countOfQueryDescriptors];

    // Fill in descriptors
    srand(36);
    for (int i = 0; i < sizeOfOneDescriptor * countOfQueryDescriptors; ++i) 
    {
		reinterpret_cast<uint8_t*>(queryDescriptors)[i] = static_cast<uint8_t>(rand());
    }
    for (int i = 0; i < sizeOfOneDescriptor * countOfTrainDescriptors; ++i) 
    {
		reinterpret_cast<uint8_t*>(trainDescriptors)[i] = static_cast<uint8_t>(rand());
    }
    
    uint64_t* deviceQueryDescriptors;
    uint64_t* deviceTrainDescriptors;
    uint32_t* deviceMatches;
    uint16_t* deviceDistances;
    cudaMalloc(&deviceQueryDescriptors, countOfQueryDescriptors * sizeOfOneDescriptor);
    cudaMalloc(&deviceTrainDescriptors, countOfTrainDescriptors * sizeOfOneDescriptor);
    cudaMalloc(&deviceMatches, countOfQueryDescriptors * sizeof(uint32_t));
    cudaMalloc(&deviceDistances, countOfQueryDescriptors * sizeof(uint16_t));

    cudaMemcpy(deviceQueryDescriptors, queryDescriptors, countOfQueryDescriptors * sizeOfOneDescriptor, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceTrainDescriptors, trainDescriptors, countOfTrainDescriptors * sizeOfOneDescriptor, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds = 0;

    cudaEventRecord(start);

    // Run the naive kernel
    for(size_t i = 0; i < numberOfRuns; i++)
    {
        BinaryMatcherNaive<<<(countOfQueryDescriptors/BLOCK_SIZE) + 1, BLOCK_SIZE>>> (deviceQueryDescriptors, deviceTrainDescriptors, countOfQueryDescriptors, countOfTrainDescriptors, deviceMatches, deviceDistances);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);

    std::cout << "Time to run naive kernel : " << std::setprecision(10) << milliseconds / numberOfRuns << " ms" << std::endl;

    cudaMemcpy(matchesNaive, deviceMatches, countOfQueryDescriptors * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(distancesNaive, deviceDistances, countOfQueryDescriptors * sizeof(uint16_t), cudaMemcpyDeviceToHost);

    // Run the kernel with shared memory usage
    cudaMemset(deviceMatches, 0, countOfQueryDescriptors * sizeof(uint32_t));
    cudaMemset(deviceDistances, 0, countOfQueryDescriptors * sizeof(uint16_t));

    // Warm-up
    for(size_t i = 0; i < numberOfRuns; i++)
    {
        BinaryMatcherWithSharedMem<<<(countOfQueryDescriptors/BLOCK_SIZE) + 1, BLOCK_SIZE>>> (reinterpret_cast<uint32_t*>(deviceQueryDescriptors), reinterpret_cast<uint32_t*>(deviceTrainDescriptors), countOfQueryDescriptors, countOfTrainDescriptors, deviceMatches, deviceDistances);
        cudaDeviceSynchronize();
    }

    cudaEventRecord(start);

    for(size_t i = 0; i < numberOfRuns; i++)
    {
        BinaryMatcherWithSharedMem<<<(countOfQueryDescriptors/BLOCK_SIZE) + 1, BLOCK_SIZE>>> (reinterpret_cast<uint32_t*>(deviceQueryDescriptors), reinterpret_cast<uint32_t*>(deviceTrainDescriptors), countOfQueryDescriptors, countOfTrainDescriptors, deviceMatches, deviceDistances);
        cudaDeviceSynchronize();
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    std::cout << "Time to run shared memory kernel : " << std::setprecision(10) << milliseconds / numberOfRuns << " ms" << std::endl;

    cudaMemcpy(matchesSharedMem, deviceMatches, countOfQueryDescriptors * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(distancesSharedMem, deviceDistances, countOfQueryDescriptors * sizeof(uint16_t), cudaMemcpyDeviceToHost);

    // Run the kernel with shared memory usage and 64-bit computation
    cudaMemset(deviceMatches, 0, countOfQueryDescriptors * sizeof(uint32_t));
    cudaMemset(deviceDistances, 0, countOfQueryDescriptors * sizeof(uint16_t));

    // Warm-up
    for(size_t i = 0; i < numberOfRuns; i++)
    {
        BinaryMatcherWithSharedMem64Bit<<<(countOfQueryDescriptors/BLOCK_SIZE) + 1, BLOCK_SIZE>>> (deviceQueryDescriptors, deviceTrainDescriptors, countOfQueryDescriptors, countOfTrainDescriptors, deviceMatches, deviceDistances);
        cudaDeviceSynchronize();
    }

    cudaEventRecord(start);

    for(size_t i = 0; i < numberOfRuns; i++)
    {
        BinaryMatcherWithSharedMem64Bit<<<(countOfQueryDescriptors/BLOCK_SIZE) + 1, BLOCK_SIZE>>> (deviceQueryDescriptors, deviceTrainDescriptors, countOfQueryDescriptors, countOfTrainDescriptors, deviceMatches, deviceDistances);
        cudaDeviceSynchronize();
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    std::cout << "Time to run shared memory 64 Bit kernel : " << std::setprecision(10) << milliseconds / numberOfRuns << " ms" << std::endl;

    cudaMemcpy(matchesSharedMem64Bit, deviceMatches, countOfQueryDescriptors * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(distancesSharedMem64Bit, deviceDistances, countOfQueryDescriptors * sizeof(uint16_t), cudaMemcpyDeviceToHost);

    // Run the kernel with shared memory usage and 64-bit computation and transposition
    cudaMemset(deviceMatches, 0, countOfQueryDescriptors * sizeof(uint32_t));
    cudaMemset(deviceDistances, 0, countOfQueryDescriptors * sizeof(uint16_t));

    // Warm-up
    for(size_t i = 0; i < numberOfRuns; i++)
    {
        BinaryMatcherWithSharedMem64BitTranspose<<<(countOfQueryDescriptors/BLOCK_SIZE) + 1, BLOCK_SIZE>>> (deviceQueryDescriptors, deviceTrainDescriptors, countOfQueryDescriptors, countOfTrainDescriptors, deviceMatches, deviceDistances);
        cudaDeviceSynchronize();
    }

    cudaEventRecord(start);

    for(size_t i = 0; i < numberOfRuns; i++)
    {
        BinaryMatcherWithSharedMem64BitTranspose<<<(countOfQueryDescriptors/BLOCK_SIZE) + 1, BLOCK_SIZE>>> (deviceQueryDescriptors, deviceTrainDescriptors, countOfQueryDescriptors, countOfTrainDescriptors, deviceMatches, deviceDistances);
        cudaDeviceSynchronize();
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    std::cout << "Time to run shared memory 64 Bit Transposition kernel : " << std::setprecision(10) << milliseconds / numberOfRuns << " ms" << std::endl;

    cudaMemcpy(matchesSharedMem64BitTranspose, deviceMatches, countOfQueryDescriptors * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(distancesSharedMem64BitTranspose, deviceDistances, countOfQueryDescriptors * sizeof(uint16_t), cudaMemcpyDeviceToHost);

    // Run OpenCV based matcher
    cv::Ptr<cv::cuda::DescriptorMatcher> openCVGPUMatcher = cv::cuda::DescriptorMatcher::createBFMatcher(cv::NORM_HAMMING);
    cv::Ptr<cv::DescriptorMatcher> openCVMatcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::MatcherType::BRUTEFORCE_HAMMING);

    cv::cuda::GpuMat queryGPUMat(countOfQueryDescriptors, sizeOfOneDescriptor, CV_8U, deviceQueryDescriptors);
    cv::cuda::GpuMat trainGPUMat(countOfTrainDescriptors, sizeOfOneDescriptor, CV_8U, deviceTrainDescriptors);
    cv::Mat queryMat(countOfQueryDescriptors, sizeOfOneDescriptor, CV_8U, queryDescriptors);
    cv::Mat trainMat(countOfTrainDescriptors, sizeOfOneDescriptor, CV_8U, trainDescriptors);
    std::vector<cv::DMatch> openCVGPUMatches;
    std::vector<cv::DMatch> openCVCPUMatches;

    // Warm-up
    for(size_t i = 0; i < numberOfRuns; i++)
    {
        openCVGPUMatcher->match(queryGPUMat, trainGPUMat, openCVGPUMatches);
    }

    std::chrono::high_resolution_clock::time_point startTime = std::chrono::high_resolution_clock::now();

    for(size_t i = 0; i < numberOfRuns; i++)
    {
        openCVGPUMatcher->match(queryGPUMat, trainGPUMat, openCVGPUMatches);
    }

    std::chrono::high_resolution_clock::time_point endTime = std::chrono::high_resolution_clock::now();
    double sec = static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(endTime - startTime).count()) * 1e-9 / static_cast<double>(numberOfRuns);
	std::cout << "openCVMatcher ran in  "  << sec * 1e3 << " ms" << std::endl;

    openCVMatcher->match(queryMat, trainMat, openCVCPUMatches);

    // Compare results with ground-truth
    for(int i = 0; i < countOfQueryDescriptors; i++)
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

        if(matchesSharedMem64BitTranspose[i] != openCVCPUMatches[i].trainIdx)
        {
            std::cout   << "queryIdx = " << i
                        << " matchesSharedMem64BitTranspose[i] := " << matchesSharedMem64BitTranspose[i] 
                        << " distance := " << distancesSharedMem64BitTranspose[i] 
                        << " and openCVCPUMatches[i].trainIdx = " << openCVCPUMatches[i].trainIdx
                        << " distance = " << openCVCPUMatches[i].distance
                        << " do not match!" << std::endl;
        }
    }

    return 0;
}