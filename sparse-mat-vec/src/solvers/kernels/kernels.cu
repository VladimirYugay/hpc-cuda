#include "kernels.h"
#include <iostream>
#include <map>
#include <sstream>

#define WARP_SIZE 32

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

void checkCublasStatus(cublasStatus_t status, const std::string &file, int line) {
    static std::map<cublasStatus_t, std::string> cublasErrorMap{
        {CUBLAS_STATUS_SUCCESS, "CUBLAS_STATUS_SUCCESS"},
        {CUBLAS_STATUS_NOT_INITIALIZED, "CUBLAS_STATUS_NOT_INITIALIZED"},
        {CUBLAS_STATUS_ALLOC_FAILED, "CUBLAS_STATUS_ALLOC_FAILED"},
        {CUBLAS_STATUS_INVALID_VALUE, "CUBLAS_STATUS_INVALID_VALUE"},
        {CUBLAS_STATUS_ARCH_MISMATCH, "CUBLAS_STATUS_ARCH_MISMATCH"},
        {CUBLAS_STATUS_MAPPING_ERROR, "CUBLAS_STATUS_MAPPING_ERROR"},
        {CUBLAS_STATUS_EXECUTION_FAILED, "CUBLAS_STATUS_EXECUTION_FAILED"},
        {CUBLAS_STATUS_INTERNAL_ERROR, "CUBLAS_STATUS_INTERNAL_ERROR"}};

    if (status == CUBLAS_STATUS_SUCCESS) {
        return;
    } else {
        std::stringstream stream;
        stream << file << ", line " << line << ": ";
        if (cublasErrorMap.find(status) != cublasErrorMap.end()) {
            stream << "cublas returned with error: " << cublasErrorMap[status];
        } else {
            stream << "cublas returned with unknown error";
        }
        throw std::runtime_error(stream.str());
    }
}

void checkCusparseStatus(cusparseStatus_t status, const std::string &file, int line) {
    static std::map<cusparseStatus_t, std::string> cusparseErrorMap{
        {CUSPARSE_STATUS_SUCCESS, "CUSPARSE_STATUS_SUCCESS"},
        {CUSPARSE_STATUS_NOT_INITIALIZED, "CUSPARSE_STATUS_NOT_INITIALIZED"},
        {CUSPARSE_STATUS_ALLOC_FAILED, "CUSPARSE_STATUS_ALLOC_FAILED"},
        {CUSPARSE_STATUS_INVALID_VALUE, "CUSPARSE_STATUS_INVALID_VALUE"},
        {CUSPARSE_STATUS_ARCH_MISMATCH, "CUSPARSE_STATUS_ARCH_MISMATCH"},
        {CUSPARSE_STATUS_MAPPING_ERROR, "CUSPARSE_STATUS_MAPPING_ERROR"},
        {CUSPARSE_STATUS_EXECUTION_FAILED, "CUSPARSE_STATUS_EXECUTION_FAILED"},
        {CUSPARSE_STATUS_INTERNAL_ERROR, "CUSPARSE_STATUS_INTERNAL_ERROR"},
        {CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED, "CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED"}};

    if (status == CUSPARSE_STATUS_SUCCESS) {
        return;
    } else {
        std::stringstream stream;
        stream << file << ", line " << line << ": ";
        if (cusparseErrorMap.find(status) != cusparseErrorMap.end()) {
            stream << "cusparse returned with error: " << cusparseErrorMap[status];
        } else {
            stream << "cusparse returned with unknown error";
        }
        throw std::runtime_error(stream.str());
    }
}
} // namespace err


std::string getDeviceName() {
    int deviceId{-1};
    cudaGetDevice(&deviceId);

    cudaDeviceProp devProp{};
    cudaGetDeviceProperties(&devProp, deviceId);
    std::stringstream stream;

    stream << devProp.name << ", Compute Capability: " << devProp.major << '.' << devProp.minor;
    return stream.str();
}

size_t get1DGrid(size_t blockSize, size_t size) {
    return (size + blockSize - 1) / blockSize;
}


//--------------------------------------------------------------------------------------------------
__global__ void kernel_csrMatVecMult(float *y, const DevCsrMatrix matrix, const float *x) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < matrix.numRows){
        float dotProduct = 0.f;
        for (int i = matrix.start[row]; i < matrix.start[row + 1]; i++){
            dotProduct += matrix.values[i] * x[matrix.indices[i]];
        }
        y[row] = dotProduct;
    }
}


template <int TILE_SIZE>
__global__ void kernel_csrMatVecMult_vectorized(float *y, const DevCsrMatrix matrix, const float *x) {
    
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int wardId = row / WARP_SIZE;
    int localId = wardId % WARP_SIZE;
    __shared__ float warpValues[TILE_SIZE];
    
    if (row < matrix.numRows){
        for (int i = matrix.start[row] + localId; i < matrix.start[row + 1]; i += WARP_SIZE){
            warpValues[threadIdx.x] += matrix.values[i] * x[matrix.indices[i]];
        }
        __syncthreads();

        // fan in all the values to the first warp
        for (int offset = 1; offset < blockDim.x; offset *= 2){
            if (localId % (2 * offset) == 0){
                warpValues[localId] += warpValues[localId + offset];
            }
            __syncthreads();
        }

        if (localId == 0){
            y[row] += warpValues[row];
        }
    } 


}


void launch_csrMatVecMult(float *y, const DevCsrMatrix matrix, const float *x,
                       const ExecutionMode mode) {
    constexpr int TILE_SIZE = 64;
    switch (mode) {
        case ExecutionMode::PAGERANK: {
            // #threads = #rows (= N)
            dim3 grid(get1DGrid(TILE_SIZE, matrix.numRows), 1, 1);
            dim3 block(TILE_SIZE, 1, 1);
            kernel_csrMatVecMult<<<grid, block>>>(y, matrix, x);
            break;
        }
        case ExecutionMode::PAGERANK_VECTORIZED: {
            // # each row is processed by a single thread in parallel
            dim3 grid(get1DGrid(TILE_SIZE, matrix.numRows * WARP_SIZE), 1, 1);
            dim3 block(TILE_SIZE, 1, 1);
            kernel_csrMatVecMult_vectorized<TILE_SIZE><<<grid, block>>>(y, matrix, x);
            break;
        }
        default: {
            std::stringstream stream;
            stream << "Unknown execution mode #(" << mode << ") for page rank solver";
            throw std::runtime_error(stream.str());
        }
    }
    CHECK_ERR;
}


//--------------------------------------------------------------------------------------------------
__global__ void kernel_ellMatVecMult(float *y, const DevEllMatrix matrix, const float *x) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < matrix.numRows){
        float dotProduct = 0.f;
        for (int i = 0; i < matrix.numColsPerRow; i++){
            int idx = i * matrix.numRows + row;
            dotProduct += x[matrix.indices[idx]] * matrix.values[idx];
        }
        y[row] = dotProduct;
    }
}


void launch_ellMatVecMult(float *y, const DevEllMatrix matrix, const float *x) {
    // thread per row 
    constexpr int TILE_SIZE = 64;
    dim3 grid(get1DGrid(TILE_SIZE, matrix.numRows), 1, 1);
    dim3 block(TILE_SIZE, 1, 1);

    kernel_ellMatVecMult<<<grid, block>>>(y, matrix, x);
    CHECK_ERR;
}


//--------------------------------------------------------------------------------------------------
__global__ void kernel_bandMatVecMult(float *y, const DevBandMatrix matrix, const float *x) {
    int row = threadIdx.x + blockIdx.x * blockDim.x;
    if (row < matrix.numRows){
        float dotProduct = 0.f;
        for (int diag = 0; diag < (2 * matrix.halfSize + 1); diag++){
            int idx = diag * matrix.numRows + row; 
            int offset = row - matrix.halfSize + diag;
            if (offset >= 0){
                dotProduct += matrix.values[idx] * x[offset];
            }
        }
        y[row] = dotProduct;
    }
}


void launch_bandMatVecMult(float *y, const DevBandMatrix matrix, const float *x) {
    //#threads = #rows (= N)
    constexpr int TILE_SIZE = 32;
    dim3 grid(get1DGrid(TILE_SIZE, matrix.numRows), 1, 1);
    dim3 block(TILE_SIZE, 1, 1);

    kernel_bandMatVecMult<<<grid, block>>>(y, matrix, x);
    CHECK_ERR;
}