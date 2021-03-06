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
    adjustedSize = getNearestPow2Number(realVectorSize);
    cudaMalloc(&vectorValues, realVectorSize * sizeof(float));
    cudaMalloc(&swapVectorValues, adjustedSize * sizeof(float));
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
        value = fmax(value, __shfl_down_sync(fullMask, value, offset));
    }

    if (threadIdx.x == 0) {
        to[blockIdx.x] = value;
    }
}


float Reducer::reduceMax(const float *values) {
    cudaMemcpy(vectorValues, values, realVectorSize * sizeof(float), cudaMemcpyHostToDevice);

    constexpr int WARP_SIZE = 32;
    dim3 block(WARP_SIZE, 1, 1);

    dim3 grid(get1DGrid(WARP_SIZE, adjustedSize), 1, 1);
    size_t swapCounter = 0;
    
    kernel_reduceMax<<<grid, block>>>(swapVectorValues, vectorValues, realVectorSize);
    std::swap(vectorValues, swapVectorValues);
    swapCounter++;
    for (int reducedSize = adjustedSize / WARP_SIZE; reducedSize > 0; reducedSize /= WARP_SIZE){
        kernel_reduceMax<<<grid, block>>>(swapVectorValues, vectorValues, reducedSize);
        std::swap(vectorValues, swapVectorValues);
        swapCounter++;
    }

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
        float value = dataField[getLinearId(idx, idy, size1D)];
        float t1 = 4.0f * (dataField[getLinearId(idx - 1, idy, size1D)] +
                                    dataField[getLinearId(idx + 1, idy, size1D)] +
                                    dataField[getLinearId(idx, idy - 1, size1D)] +
                                    dataField[getLinearId(idx, idy + 1, size1D)]);

        float t2 = dataField[getLinearId(idx + 1, idy + 1, size1D)] +
                      dataField[getLinearId(idx - 1, idy + 1, size1D)] +
                      dataField[getLinearId(idx + 1, idy - 1, size1D)] +
                      dataField[getLinearId(idx - 1, idy - 1, size1D)];    
        
        swapField[getLinearId(idx, idy, size1D)] = value + (params.factor * (t1 + t2 - 20.0f * value) * params.invDhSquare) / 6.0f;
    }
}


void launch_gpuSimple(float *swapField, const float *dataField, const Settings &settings,
                      const Params &params) {
    dim3 block(settings.blockSizeX, settings.blockSizeY, 1);

    dim3 grid(get1DGrid(block.x, settings.num1dGridPoints - 2),
              get1DGrid(block.y, settings.num1dGridPoints - 2), 1);

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
        patch[getShrLinearId(threadIdx.x, threadIdx.y, blockDim.x)] =
            dataField[getRealLinearId(idx, idy, size1D)];
    }
    __syncthreads();

    const size_t realLinearId = getRealLinearId(idx, idy, size1D);
    const float value = patch[getShrLinearId(threadIdx.x, threadIdx.y, blockDim.x)];
    if ((idx < (size1D - 1)) && (idy < (size1D - 1))) {

        if ((threadIdx.x > 0) && (threadIdx.y > 0) && (threadIdx.x < (blockDim.x - 1)) &&
            (threadIdx.y < (blockDim.y - 1))) {

            float t1 = 4.0f * (patch[getShrLinearId(threadIdx.x - 1, threadIdx.y, blockDim.x)] +
                                  patch[getShrLinearId(threadIdx.x + 1, threadIdx.y, blockDim.x)] +
                                  patch[getShrLinearId(threadIdx.x, threadIdx.y - 1, blockDim.x)] +
                                  patch[getShrLinearId(threadIdx.x, threadIdx.y + 1, blockDim.x)]);

            float t2 = patch[getShrLinearId(threadIdx.x + 1, threadIdx.y + 1, blockDim.x)] +
                          patch[getShrLinearId(threadIdx.x - 1, threadIdx.y + 1, blockDim.x)] +
                          patch[getShrLinearId(threadIdx.x + 1, threadIdx.y - 1, blockDim.x)] +
                          patch[getShrLinearId(threadIdx.x - 1, threadIdx.y - 1, blockDim.x)];

            swapField[realLinearId] =
                value +
                (params.factor * (t1 + t2 - 20.0f * value) * params.invDhSquare) / 6.0f;
        }
    }
}


void launch_gpuWithShrMem(float *swapField, const float *dataField, const Settings &settings,
                          const Params &params) {

    dim3 block(settings.blockSizeX, settings.blockSizeY, 1);
    dim3 grid(get1DGrid(block.x - 2, settings.num1dGridPoints),
              get1DGrid(block.y - 2, settings.num1dGridPoints), 1);

    const size_t shrMemSize = block.x * block.y * sizeof(float);
    kernel_gpuWithShrMem<<<grid, block, shrMemSize>>>(swapField, dataField, params,
                                                      settings.num1dGridPoints);
    CHECK_ERR;
}
