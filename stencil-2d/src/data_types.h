#ifndef STENCIL_DATA_TYPES_H
#define STENCIL_DATA_TYPES_H

#include <cstdlib>
#include <limits>

enum ExecutionMode { CPU, GPU_GLOB_MEM, GPU_SHR_MEM };

struct Settings {
    size_t num1dGridPoints{10};
    size_t numIterations{std::numeric_limits<size_t>::max()};
    size_t stepsPerPrint{500};
    ExecutionMode mode{ExecutionMode::CPU};
    size_t blockSizeX{32};
    size_t blockSizeY{4};
    float conductivity{0.1};
    float endTime{1.0};
    float eps{1e-6};
    bool isVerbose{true};
};

struct Params {
    float invDhSquare{};
    float factor{};
    float dtStable{};
};


#endif // STENCIL_DATA_TYPES_H
