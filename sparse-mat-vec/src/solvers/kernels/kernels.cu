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
  // TODO: H3.1 implement mat-vec multiplication
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
          // TODO: H3.1 define grid/block size
            dim3 grid(1, 1, 1);
            dim3 block(1, 1, 1);
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
  // TODO: T4.1a
}


void launch_ellMatVecMult(float *y, const DevEllMatrix matrix, const float *x) {
  // TODO: T4.1a
    dim3 grid(1, 1, 1);
    dim3 block(1, 1, 1);

    kernel_ellMatVecMult<<<grid, block>>>(y, matrix, x);
    CHECK_ERR;
}


//--------------------------------------------------------------------------------------------------
__global__ void kernel_bandMatVecMult(float *y, const DevBandMatrix matrix, const float *x) {
  // TODO: H5.1
}


void launch_bandMatVecMult(float *y, const DevBandMatrix matrix, const float *x) {
    // TODO: H5.1
    //#threads = #rows (= N)
    dim3 grid(1, 1, 1);
    dim3 block(1, 1, 1);

    kernel_bandMatVecMult<<<grid, block>>>(y, matrix, x);
    CHECK_ERR;
}