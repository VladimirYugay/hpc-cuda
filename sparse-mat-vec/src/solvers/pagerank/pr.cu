#include "pr.h"
#include "kernels/kernels.h"
#include <iostream>
#include <limits>
#include <chrono>

using namespace std::chrono;

namespace pr {

void printSolution(const std::vector<float> &x);

double pageRank(CsrMatrix matrix, ExecutionMode mode, Settings settings) {

    std::vector<int> ni(matrix.numRows, 0);
    // count outgoing links. matrix entry i, j describes
    // incoming links to i from j, so we need a column sum
    for (int i = 0; i < matrix.nnz; ++i) {
        ni[matrix.indices[i]] += 1;
    }

    for (int i = 0; i < matrix.numRows; ++i) {
        if (ni[i] == 0) {
            std::cout << "Warning: Column " << i << " sum is zero, non-stochastic matrix!\n";
            break;
        }
    }
    std::cout << std::endl;

    // weight columns by nr of outgoing links (previously calculated)
    for (int i = 0; i < matrix.nnz; ++i) {
        matrix.values[i] /= ni[matrix.indices[i]];
    }

    // init x vector with 1/N
    std::vector<float> x(matrix.numRows);
    for (int i = 0; i < matrix.numRows; ++i) {
        x[i] = 1.0 / matrix.numRows;
    }
    // init x vector with 0
    std::vector<float> y(matrix.numRows, 0.0f);

    high_resolution_clock::time_point start = high_resolution_clock::now();

    DevCsrMatrix devMatrix;
    devMatrix.numRows = matrix.numRows;
    devMatrix.nnz = matrix.nnz;
    // allocate and copy data for: devMatrix, devX, and devY
    // clang-format off
    cudaMalloc(&devMatrix.start, (matrix.numRows + 1) * sizeof(int)); CHECK_ERR;
    cudaMemcpy(devMatrix.start, matrix.start.data(), (matrix.numRows + 1) * sizeof(int),
               cudaMemcpyHostToDevice); CHECK_ERR;

    cudaMalloc(&devMatrix.indices, matrix.nnz * sizeof(int)); CHECK_ERR;
    cudaMemcpy(devMatrix.indices, matrix.indices.data(), matrix.nnz * sizeof(int),
               cudaMemcpyHostToDevice); CHECK_ERR;

    cudaMalloc(&devMatrix.values, (matrix.nnz + 1) * sizeof(float)); CHECK_ERR;
    cudaMemcpy(devMatrix.values, matrix.values.data(), matrix.nnz * sizeof(float),
               cudaMemcpyHostToDevice); CHECK_ERR;

    float *devX{nullptr};
    cudaMalloc(&devX, matrix.numRows * sizeof(float)); CHECK_ERR;
    cudaMemcpy(devX, x.data(), matrix.numRows * sizeof(float), cudaMemcpyHostToDevice); CHECK_ERR;

    float *devY{nullptr};
    cudaMalloc(&devY, matrix.numRows * sizeof(float)); CHECK_ERR;
    cudaMemcpy(devY, y.data(), matrix.numRows * sizeof(float), cudaMemcpyHostToDevice); CHECK_ERR;
    // clang-format on

    // main compute-loop
    float err = std::numeric_limits<float>::max();
    for (int counter = 1; ((err > settings.epsilon) && (settings.numIterations > counter)); counter++) {
        // compute y = B x
        launch_csrMatVecMult(devY, devMatrix, devX, mode);

        cudaMemcpy(const_cast<float *>(y.data()), devY, matrix.numRows * sizeof(float),
                   cudaMemcpyDeviceToHost); CHECK_ERR;
        cudaMemcpy(const_cast<float *>(x.data()), devX, matrix.numRows * sizeof(float),
                   cudaMemcpyDeviceToHost); CHECK_ERR;
        err = 0.0f;
        float sum = 0.0f;
        for (int j = 0; j < matrix.numRows; ++j) {
            // do regularization x' = a y + (1 - a) / N * e
            float newX = settings.alpha * y[j] + (1.0f - settings.alpha) * 1.0f / matrix.numRows;

            // calculate error
            err += std::fabs(x[j] - newX);

            // replace x with x'
            x[j] = newX;
            sum += x[j];
        }
        cudaMemcpy(devX, x.data(), matrix.numRows * sizeof(float), cudaMemcpyHostToDevice);
        CHECK_ERR;
        std::cout << "Iterations = " << counter << ", err = " << err << ", sum = " << sum
                  << std::endl;
    }

    cudaMemcpy(const_cast<float *>(x.data()), devX, matrix.numRows * sizeof(float),
               cudaMemcpyDeviceToHost); CHECK_ERR;
    printSolution(x);

    // deallocate memory occupied with devMatrix, devX, and devY
    // clang-format off
    cudaFree(devMatrix.start); CHECK_ERR;
    cudaFree(devMatrix.indices); CHECK_ERR;
    cudaFree(devMatrix.values); CHECK_ERR;
    cudaFree(devX); CHECK_ERR;
    cudaFree(devY); CHECK_ERR;
    // clang-format on

    high_resolution_clock::time_point end = high_resolution_clock::now();
    return duration_cast<duration<double>>(end - start).count();
}


void printSolution(const std::vector<float> &x) {
    int i_max = 0;
    float x_max = 0.0f;

    for (int i = 0; i < x.size(); ++i) {
        if (x[i] > x_max) {
            i_max = i;
            x_max = x[i];
        }

        constexpr int LIMIT = 10;
        if (i < LIMIT) {
            std::cout << "x_" << i << " = " << x[i] << std::endl;
        }
    }
    std::cout << "\nMaximum: x_" << i_max << " = " << x_max << std::endl;
}
} // namespace pr