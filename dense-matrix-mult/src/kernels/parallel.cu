#include "kernels.h"
#include "util.hpp"
#include <math.h>
#include <iostream>

namespace gpu {


size_t get1DGrid(size_t blockSize, size_t matrixSize) {
    return matrixSize % blockSize == 0 ? matrixSize / blockSize : matrixSize / blockSize + 1; 
}


//--------------------------------------------------------------------------------------------------
__global__ void kernel_matrixMultGlobal(const float *devA, const float *devB, float *devC,
                                        const int size) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if ((col < size) && (row < size)) {
        float dotProduct = 0.0f;
        for (int i = 0; i < size; i++) {
            dotProduct += devA[i * size + row] * devB[col * size + i];
        }
        devC[col * size + row] += dotProduct;
    }
}

void executeMatrixMultGlobal(dim3 dimBlock, dim3 dimGrid, float *Ad, float *Bd, float *Cd,
                             const Configuration &config) {
    for (int i = 0; i < config.numRepeats; ++i) {
        kernel_matrixMultGlobal<<<dimGrid, dimBlock>>>(Ad, Bd, Cd, config.matrixSize);
    }
    CHECK_ERR;
}


//--------------------------------------------------------------------------------------------------
template <int TILE_SIZE>
__global__ void kernel_matrixMultTiled(const float *__restrict__ devA,
                                       const float *__restrict__ devB, float *__restrict__ devC,
                                       const size_t size) {
    
    int tx = threadIdx.x;
    int ty = threadIdx.y; 
    int bx = blockIdx.x;
    int by = blockIdx.y; 

    int col = bx * blockDim.x + tx;
    int row = by * blockDim.y + ty;

    // col-major format
    __shared__ float sharedA[TILE_SIZE][TILE_SIZE];
    __shared__ float sharedB[TILE_SIZE][TILE_SIZE];

    if ((col < size) && (row < size)) {

        float dotProduct = 0;

        for (int bid = 0; bid < size / TILE_SIZE; bid++) {
            
            sharedA[tx][ty] = devA[bid * size * TILE_SIZE + tx * size + by * TILE_SIZE + ty];
            sharedB[tx][ty] = devB[bx * size * TILE_SIZE + tx * size + bid * TILE_SIZE + ty];
            __syncthreads();

            for (int i = 0; i < TILE_SIZE; i++) {
                dotProduct += sharedA[i][ty] * sharedB[tx][i];
            }
            __syncthreads();
        }
        devC[col * size + row] += dotProduct;

    }
}


void executeMatrixMultTiled(dim3 dimBlock, dim3 dimGrid, float *Ad, float *Bd, float *Cd,
                            const Configuration &config) {
    switch (config.tileSize) {
        case 2:
            for (int i = 0; i < config.numRepeats; ++i) {
                kernel_matrixMultTiled<2><<<dimGrid, dimBlock>>>(Ad, Bd, Cd, config.matrixSize);
            }
            break;        
        case 4:
            for (int i = 0; i < config.numRepeats; ++i) {
                kernel_matrixMultTiled<4><<<dimGrid, dimBlock>>>(Ad, Bd, Cd, config.matrixSize);
            }
            break;
        case 8:
            for (int i = 0; i < config.numRepeats; ++i) {
                kernel_matrixMultTiled<8><<<dimGrid, dimBlock>>>(Ad, Bd, Cd, config.matrixSize);
            }
            break;
        case 16:
            for (int i = 0; i < config.numRepeats; ++i) {
                kernel_matrixMultTiled<16><<<dimGrid, dimBlock>>>(Ad, Bd, Cd, config.matrixSize);
            }
            break;
        case 32:
            for (int i = 0; i < config.numRepeats; ++i) {
                kernel_matrixMultTiled<32><<<dimGrid, dimBlock>>>(Ad, Bd, Cd, config.matrixSize);
            }
            break;
    }
    CHECK_ERR;
}


//--------------------------------------------------------------------------------------------------
template <int TILE_SIZE>
__global__ void kernel_matrixMultCoalesced(const float *__restrict__ devA,
                                           const float *__restrict__ devB, float *__restrict__ devC,
                                           const size_t size) {
    int tx = threadIdx.x;
    int ty = threadIdx.y; 
    int bx = blockIdx.x;
    int by = blockIdx.y; 

    int col = bx * blockDim.x + tx;
    int row = by * blockDim.y + ty;

    // col-major format
    __shared__ float sharedA[TILE_SIZE][TILE_SIZE];
    __shared__ float sharedB[TILE_SIZE][TILE_SIZE];

    if ((col < size) && (row < size)) {

        float dotProduct = 0;

        for (int bid = 0; bid < size / TILE_SIZE; bid++) {
            
            sharedA[ty][tx] = devA[bid * size * TILE_SIZE + ty * size + by * TILE_SIZE + tx];
            sharedB[ty][tx] = devB[bx * size * TILE_SIZE + ty * size + bid * TILE_SIZE + tx];
            __syncthreads();

            for (int i = 0; i < TILE_SIZE; i++) {
                dotProduct += sharedA[i][ty] * sharedB[tx][i];
            }
            __syncthreads();
        }
        devC[col * size + row] += dotProduct;

    }
}


void executeMatrixMultCoalesced(dim3 dimBlock, dim3 dimGrid, float *Ad, float *Bd, float *Cd,
                                const Configuration &config) {
    switch (config.tileSize) {
        case 2:
            for (int i = 0; i < config.numRepeats; ++i) {
                kernel_matrixMultCoalesced<2><<<dimGrid, dimBlock>>>(Ad, Bd, Cd, config.matrixSize);
            }
            break;        
        case 4:
            for (int i = 0; i < config.numRepeats; ++i) {
                kernel_matrixMultCoalesced<4><<<dimGrid, dimBlock>>>(Ad, Bd, Cd, config.matrixSize);
            }
            break;
        case 8:
            for (int i = 0; i < config.numRepeats; ++i) {
                kernel_matrixMultCoalesced<8><<<dimGrid, dimBlock>>>(Ad, Bd, Cd, config.matrixSize);
            }
            break;
        case 16:
            for (int i = 0; i < config.numRepeats; ++i) {
                kernel_matrixMultCoalesced<16>
                    <<<dimGrid, dimBlock>>>(Ad, Bd, Cd, config.matrixSize);
            }
            break;
        case 32:
            for (int i = 0; i < config.numRepeats; ++i) {
                kernel_matrixMultCoalesced<32>
                    <<<dimGrid, dimBlock>>>(Ad, Bd, Cd, config.matrixSize);
            }
            break;
    }
    CHECK_ERR;
}


//--------------------------------------------------------------------------------------------------
__global__ void kernel_matrixMultCoalescedDym(const float *__restrict__ devA,
                                              const float *__restrict__ devB,
                                              float *__restrict__ devC, const size_t size) {
  // TODO: complete function
}


void executeMatrixMultCoalescedDym(dim3 dimBlock, dim3 dimGrid, float *Ad, float *Bd, float *Cd,
                                   const Configuration &config) {
    const size_t shrMemSize = 2 * config.tileSize * config.tileSize * sizeof(float);
    for (int i = 0; i < config.numRepeats; ++i) {
        kernel_matrixMultCoalescedDym<<<dimGrid, dimBlock, shrMemSize>>>(Ad, Bd, Cd,
                                                                         config.matrixSize);
    }
    CHECK_ERR;
}


//--------------------------------------------------------------------------------------------------
template <int TILE_SIZE>
__global__ void kernel_matrixMultOverlapped(const float *__restrict__ devA,
                                            const float *__restrict__ devB,
                                            float *__restrict__ devC, const size_t size) {
    int tx = threadIdx.x;
    int ty = threadIdx.y; 
    int bx = blockIdx.x;
    int by = blockIdx.y; 

    int col = bx * blockDim.x + tx;
    int row = by * blockDim.y + ty;

    // col-major format
    __shared__ float sharedA[TILE_SIZE][TILE_SIZE];
    __shared__ float sharedB[TILE_SIZE][TILE_SIZE];

    if ((col < size) && (row < size)) {

        float dotProduct = 0;
        float regA = 0.f, regB = 0.f;
        regA = devA[ty * size + by * TILE_SIZE + tx];
        regB = devB[bx * size * TILE_SIZE + ty * size + tx];

        for (int bid = 0; bid < size / TILE_SIZE - 1; bid++) {
            
            sharedA[ty][tx] = regA;
            sharedB[ty][tx] = regB;
            __syncthreads();

            regA = devA[(bid + 1) * size * TILE_SIZE + ty * size + by * TILE_SIZE + tx];
            regB = devB[bx * size * TILE_SIZE + ty * size + (bid + 1) * TILE_SIZE + tx];
            for (int i = 0; i < TILE_SIZE; i++) {
                dotProduct += sharedA[i][ty] * sharedB[tx][i];
            }
            __syncthreads();
        }

        int bid = size / TILE_SIZE - 1;
        sharedA[ty][tx] = devA[bid * size * TILE_SIZE + ty * size + by * TILE_SIZE + tx];
        sharedB[ty][tx] = devB[bx * size * TILE_SIZE + ty * size + bid * TILE_SIZE + tx];
        __syncthreads();
        
        for (int i = 0; i < TILE_SIZE; i++) {
            dotProduct += sharedA[i][ty] * sharedB[tx][i];
        }        

        devC[col * size + row] += dotProduct;

    }
}


void executeMatrixMultOverlapped(dim3 dimBlock, dim3 dimGrid, float *Ad, float *Bd, float *Cd,
                                 const Configuration &config) {
    switch (config.tileSize) {
        case 2:
            for (int i = 0; i < config.numRepeats; ++i) {
                kernel_matrixMultOverlapped<2>
                    <<<dimGrid, dimBlock>>>(Ad, Bd, Cd, config.matrixSize);
            }
            break;        
        case 4:
            for (int i = 0; i < config.numRepeats; ++i) {
                kernel_matrixMultOverlapped<4>
                    <<<dimGrid, dimBlock>>>(Ad, Bd, Cd, config.matrixSize);
            }
            break;
        case 8:
            for (int i = 0; i < config.numRepeats; ++i) {
                kernel_matrixMultOverlapped<8>
                    <<<dimGrid, dimBlock>>>(Ad, Bd, Cd, config.matrixSize);
            }
            break;
        case 16:
            for (int i = 0; i < config.numRepeats; ++i) {
                kernel_matrixMultOverlapped<16>
                    <<<dimGrid, dimBlock>>>(Ad, Bd, Cd, config.matrixSize);
            }
            break;
        case 32:
            for (int i = 0; i < config.numRepeats; ++i) {
                kernel_matrixMultOverlapped<32>
                    <<<dimGrid, dimBlock>>>(Ad, Bd, Cd, config.matrixSize);
            }
            break;
    }
    CHECK_ERR;
}

//--------------------------------------------------------------------------------------------------
void matrixMult(float *Ad, float *Bd, float *Cd, const Configuration &config) {
    dim3 dimBlock(config.tileSize, config.tileSize);
    const size_t Grid1D = get1DGrid(config.tileSize, config.matrixSize);
    dim3 dimGrid(Grid1D, Grid1D);

    switch (config.kernelType) {
        case KernelType::KERNEL_GLOBAL:
            executeMatrixMultGlobal(dimBlock, dimGrid, Ad, Bd, Cd, config);
            break;
        case KernelType::KERNEL_TILED:
            executeMatrixMultTiled(dimBlock, dimGrid, Ad, Bd, Cd, config);
            break;
        case KernelType::KERNEL_COALESCED:
            executeMatrixMultCoalesced(dimBlock, dimGrid, Ad, Bd, Cd, config);
            break;
        case KernelType::KERNEL_COALESCED_DYM:
            executeMatrixMultCoalescedDym(dimBlock, dimGrid, Ad, Bd, Cd, config);
            break;
        case KernelType::KERNEL_OVERLAPPED:
            executeMatrixMultOverlapped(dimBlock, dimGrid, Ad, Bd, Cd, config);
            break;
    }
    CHECK_ERR;
}
} // namespace gpu
