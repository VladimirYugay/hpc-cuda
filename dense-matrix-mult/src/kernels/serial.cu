#include "kernels.h"
#include <iostream>

namespace cpu {
void matrixMult(std::vector<float> &C, const std::vector<float> &A, const std::vector<float> &B,
                const Configuration &config) {

    const int size = config.matrixSize;
    for (int r = 0; r < config.numRepeats; ++r) {
        for (int row = 0; row < size; row++) {
            for (int col = 0; col < size; col++) {
                float dotProduct = 0;
                for (int k = 0; k < size; k++) {
                    dotProduct += A[k * size + row] * B[col * size + k];
                }
                C[col * size + row] += dotProduct;
            }
        }
    }
}
} // namespace cpu
