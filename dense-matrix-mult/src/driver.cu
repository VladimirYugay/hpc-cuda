#include "driver.h"
#include "kernels/kernels.h"
#include "util.hpp"
#include <chrono>
#include <cublas_v2.h>

float compute(std::vector<float> &C, const std::vector<float> &A, const std::vector<float> &B,
              const Configuration &config) {

    cudaDeviceReset();
    CHECK_ERR;

    float *devA{nullptr}, *devB{nullptr}, *devC{nullptr};
    {
      // Allocate matrices A, B, C on device
      cudaMalloc(&devA, A.size() * sizeof(float));
      cudaMalloc(&devB, B.size() * sizeof(float));
      cudaMalloc(&devC, C.size() * sizeof(float));
    }

    {
      // Copy the data from host to the device
      // NOTE: You may copy C as well, as it is zeroed, or cudaMemset it to zero on the device
      cudaMemcpy(devA, A.data(), A.size() * sizeof(float), cudaMemcpyHostToDevice);
      cudaMemcpy(devB, B.data(), B.size() * sizeof(float), cudaMemcpyHostToDevice);
      cudaMemcpy(devC, C.data(), C.size() * sizeof(float), cudaMemcpyHostToDevice);

    }

    cublasHandle_t handle;
    cublasCreate(&handle);

    float cpuTime{};
    cudaEvent_t startTimer{}, stopTimer{};
    cudaEventCreate(&startTimer);
    cudaEventCreate(&stopTimer);


    // Start computing
    cudaEventRecord(startTimer, 0);
    switch (config.kernelType) {
        case KernelType::KERNEL_CPU: {
            auto begin = std::chrono::high_resolution_clock::now();
            // NOTE: repeat loop is inside of cpu::matrixMult
            cpu::matrixMult(C, A, B, config);
            auto end = std::chrono::high_resolution_clock::now();
            cpuTime =
                std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(end - begin).count();
            break;
        }
        case KernelType::KERNEL_CUBLAS: {
            float alpha = 1.0f, beta = 1.0f;
            for (int i = 0; i < config.numRepeats; ++i) {
                cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, config.matrixSize, config.matrixSize,
                            config.matrixSize, &alpha, devA, config.matrixSize, devB,
                            config.matrixSize, &beta, devC, config.matrixSize);
            }
            CHECK_ERR;
            break;
        }
        default: {
            // NOTE: repeat loop is inside of gpu::matrixMult
            gpu::matrixMult(devA, devB, devC, config);
            break;
        }
    }

    cudaEventRecord(stopTimer, 0);
    cudaEventSynchronize(stopTimer);
    CHECK_ERR;
    // NOTE: cudaEventSynchronize(stopTimer) is implicit cudaDeviceSynchronize() in this context

    float gpuTime{};
    cudaEventElapsedTime(&gpuTime, startTimer, stopTimer);

    // release resources
    cublasDestroy(handle);
    cudaEventDestroy(startTimer);
    cudaEventDestroy(stopTimer);

    {
      // transfer matrix C back, from device to the host
      cudaMemcpy(const_cast<float *>(C.data()), devC, C.size() * sizeof(float), cudaMemcpyDeviceToHost);
    }

    {
      // clean gpu memory
      cudaFree(devA);
      cudaFree(devB);
      cudaFree(devC);
    }
    return (config.kernelType == KernelType::KERNEL_CPU) ? cpuTime : gpuTime;
}
