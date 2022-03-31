#include "aux.h"
#include "kernels.h"
#include <cfloat>
#include <iostream>
#include <map>
#include <sstream>


namespace err {
std::string PrevFile{};
int PrevLine{0};


void checkErr(const std::string &file, int line) {
#ifndef NDEBUG
    cudaError_t Error = cudaGetLastError();
    if (Error != cudaSuccess) {
        std::stringstream stream;
        stream << '\n'
               << file << ", line " << line << ": " << cudaGetErrorString(Error) << " (" << Error
               << ")\n";
        if (PrevLine > 0) {
            stream << "Previous CUDA call:" << '\n' << PrevFile << ", line " << PrevLine << '\n';
        }
        throw std::runtime_error(stream.str());
    }
    PrevFile = file;
    PrevLine = line;
#endif
}
} // namespace err


std::string getDeviceName() {
    int deviceId{-1};
    cudaGetDevice(&deviceId);

    cudaDeviceProp devProp{};
    cudaGetDeviceProperties(&devProp, deviceId);
    CHECK_ERR;
    std::stringstream stream;

    stream << devProp.name << ", Compute Capability: " << devProp.major << '.' << devProp.minor;
    return stream.str();
}

size_t getMaxNumThreadPerBlock() {
    int deviceId{-1};
    cudaGetDevice(&deviceId);

    cudaDeviceProp devProp{};
    cudaGetDeviceProperties(&devProp, deviceId);
    CHECK_ERR;
    return devProp.maxThreadsPerBlock;
}


size_t get1DGrid(size_t blockSize, size_t size) {
    return (size + blockSize - 1) / blockSize;
}


__device__ size_t getLinearId(long int idx, long int idy, size_t size) {
    return (idx + 1) + (idy + 1) * size;
}


__device__ size_t getRealLinearId(long int idx, long int idy, size_t size) {
    return idx + idy * size;
}
//--------------------------------------------------------------------------------------------------
void __global__ kernel_computeErr(float *errVector, const float *vector1, const float *vector2,
                                  size_t size) {
    const size_t id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < size) {
        errVector[id] = fabs(vector1[id] - vector2[id]);
    }
}

void launch_computeErr(float *errVector, const float *vector1, const float *vector2, size_t size) {
    dim3 block(256, 1, 1);
    dim3 grid(get1DGrid(block.x, size), 1, 1);
    kernel_computeErr<<<grid, block>>>(errVector, vector1, vector2, size);
    CHECK_ERR;
}

//--------------------------------------------------------------------------------------------------
Reducer::Reducer(size_t size) : realVectorSize(size) {
    // TODO: implement getNearestPow2Number function (see aux.cpp)
    adjustedSize = getNearestPow2Number(realVectorSize);
    // TODO: allocate enough memory for vectorValues and swapVectorValues
}

Reducer::~Reducer() {
    cudaFree(vectorValues);
    CHECK_ERR;
    cudaFree(swapVectorValues);
    CHECK_ERR;
}


void __global__ kernel_reduceMax(float *to, const float *from, const size_t size) {
    const size_t id = threadIdx.x + blockIdx.x * blockDim.x;
    float value = (id < size) ? from[id] : FLT_MIN;

    constexpr unsigned int fullMask = 0xffffffff;
    for (int offset = 16; offset > 0; offset /= 2) {
        // TODO:  compute value. Consider: __shfl_???_sync(fullMask, ???, ???));
        value = 0;
    }

    if (threadIdx.x == 0) {
        to[blockIdx.x] = value;
    }
}


float Reducer::reduceMax(const float *values) {
    // TODO: copy data from values to vectorValues. Use cudaMemcpy

    constexpr int WARP_SIZE = 32;
    dim3 block(WARP_SIZE, 1, 1);

    // TODO: adjust grid size
    dim3 grid(1, 1, 1);

    size_t swapCounter = 0;
    // TODO: implement grid level reduction

    float results{};
    if ((swapCounter % 2) == 0) {
        cudaMemcpy(&results, vectorValues, sizeof(float), cudaMemcpyDeviceToHost);
        CHECK_ERR;
    } else {
        cudaMemcpy(&results, swapVectorValues, sizeof(float), cudaMemcpyDeviceToHost);
        CHECK_ERR;
    }
    return results;
}


//--------------------------------------------------------------------------------------------------
__global__ void kernel_gpuSimple(float *swapField, const float *dataField, const Params params,
                                 const size_t size1D) {
    long int idx = threadIdx.x + blockIdx.x * blockDim.x;
    long int idy = threadIdx.y + blockIdx.y * blockDim.y;

    if ((idx < (size1D - 2)) && (idy < (size1D - 2))) {
        // TODO: complete this kernel. Update swapField (you can use getLinearId function)

    }
}


void launch_gpuSimple(float *swapField, const float *dataField, const Settings &settings,
                      const Params &params) {
    dim3 block(settings.blockSizeX, settings.blockSizeY, 1);

    // TODO: find grid/block distribution
    dim3 grid(1, 1, 1);

    kernel_gpuSimple<<<grid, block>>>(swapField, dataField, params, settings.num1dGridPoints);
    CHECK_ERR;
}


//--------------------------------------------------------------------------------------------------
__device__ size_t getShrLinearId(long int idx, long int idy, size_t rowSize) {
    return idx + idy * rowSize;
}


__global__ void kernel_gpuWithShrMem(float *swapField, const float *dataField, const Params params,
                                     const size_t size1D) {
    const long int idx = threadIdx.x + blockIdx.x * (blockDim.x - 2);
    const long int idy = threadIdx.y + blockIdx.y * (blockDim.y - 2);


    // load a patch (blockDim.x + 2) x (blockDim.y + 2) to the shared memory
    extern __shared__ float patch[];
    if ((idx < size1D) && (idy < size1D)) {
        // TODO: load a patch from global to shared memory
    }
    __syncthreads();

    const size_t realLinearId = getRealLinearId(idx, idy, size1D);
    const float selfValue = patch[getShrLinearId(threadIdx.x, threadIdx.y, blockDim.x)];
    // TODO: complete this kernel. Update swapField (you can use getLinearId function)
}


void launch_gpuWithShrMem(float *swapField, const float *dataField, const Settings &settings,
                          const Params &params) {

    dim3 block(settings.blockSizeX, settings.blockSizeY, 1);
    // TODO: find grid/block distribution
    dim3 grid(1, 1, 1);

    // TODO: compute the amount of shared memory
    const size_t shrMemSize = 1;
    kernel_gpuWithShrMem<<<grid, block, shrMemSize>>>(swapField, dataField, params,
                                                      settings.num1dGridPoints);
    CHECK_ERR;
}
