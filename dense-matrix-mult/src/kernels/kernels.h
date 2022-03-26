#ifndef KERNELS_H
#define KERNELS_H

#include "data_types.h"
#include "stdio.h"
#include <vector>

namespace cpu {
void matrixMult(std::vector<float> &C, const std::vector<float> &A, const std::vector<float> &B,
                const Configuration &config);
}

namespace gpu {
void matrixMult(float *Ad, float *Bd, float *Cd, const Configuration &config);
size_t get1DGrid(size_t blockSize, size_t matrixSize);
} // namespace gpu

#endif // KERNELS_H
