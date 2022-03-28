#ifndef SPARSE_DATA_TYPES_H
#define SPARSE_DATA_TYPES_H

#include <string>
#include <vector>
#include <limits>

enum ExecutionMode { PAGERANK, PAGERANK_VECTORIZED, HEAT_ELLPACK, HEAT_CUSPARSE, HEAT_BAND };


struct Configuration {
    ExecutionMode executionMode{ExecutionMode::PAGERANK};
    std::string matrixFile{""};
    size_t numGridPoints{100};
    size_t numIterations{std::numeric_limits<size_t>::max()};
    size_t stepsPerPrint{500};
};


struct CsrMatrix {
    int numRows{};
    int nnz{};
    std::vector<int> start{};
    std::vector<int> indices{};
    std::vector<float> values{};
};


struct DevCsrMatrix {
    int numRows{};
    int nnz{};
    int *start{nullptr};
    int *indices{nullptr};
    float *values{nullptr};
};


struct EllMatrix {
    EllMatrix() = default;
    explicit EllMatrix(size_t numRows, int colsPerRow)
        : numRows(numRows), numColsPerRow(colsPerRow) {
        indices.resize(numColsPerRow * numRows, 0);
        values.resize(numColsPerRow * numRows, 0.0f);
    }

    EllMatrix(const EllMatrix &other) = default;
    EllMatrix &operator=(const EllMatrix &other) = default;

    int numColsPerRow{0};
    size_t numRows{0};
    std::vector<int> indices{};
    std::vector<float> values{};
};


struct DevEllMatrix {
    int numColsPerRow{0};
    size_t numRows{0};
    int *indices{nullptr};
    float *values{nullptr};
};


struct DevBandMatrix {
  int halfSize{0};
  size_t numRows{0};
  float *values{nullptr};
};

#endif // SPARSE_DATA_TYPES_H
