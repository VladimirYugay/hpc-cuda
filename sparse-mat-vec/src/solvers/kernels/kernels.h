#ifndef SPARSE_KERNELS_HPP
#define SPARSE_KERNELS_HPP


#include "data_types.h"
#include "kernels.h"
#include <cublas_v2.h>
#include <cusparse_v2.h>
#include <cuda.h>
#include <string>

namespace err {
void checkErr(const std::string &file, int line);
void checkCublasStatus(cublasStatus_t status, const std::string &file, int line);
void checkCusparseStatus(cusparseStatus_t status, const std::string &file, int line);
} // namespace err
#define CHECK_ERR err::checkErr(__FILE__, __LINE__)
#define CUBLAS_CHECK(EXPR) err::checkCublasStatus(EXPR, __FILE__, __LINE__)
#define CUSPARSE_CHECK(EXPR) err::checkCusparseStatus(EXPR, __FILE__, __LINE__)

std::string getDeviceName();

void launch_csrMatVecMult(float *y, const DevCsrMatrix matrix, const float *x,
                       const ExecutionMode mode);

void launch_ellMatVecMult(float *y,const DevEllMatrix matrix, const float *x);

void launch_bandMatVecMult(float *y, const DevBandMatrix matrix, const float *x);


#endif // SPARSE_KERNELS_HPP
