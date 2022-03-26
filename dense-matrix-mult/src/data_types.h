#ifndef DATA_TYPES_H
#define DATA_TYPES_H

enum class KernelType : int {
    KERNEL_CPU,
    KERNEL_GLOBAL,
    KERNEL_TILED,
    KERNEL_COALESCED,
    KERNEL_COALESCED_DYM,
    KERNEL_OVERLAPPED,
    KERNEL_CUBLAS
};

struct Configuration {
    bool printMatrix{false};
    bool printInfo{false};
    int tileSize{2};
    int matrixSize{4};
    int numRepeats{1};
    KernelType kernelType{KernelType::KERNEL_CPU};
};

#endif // DATA_TYPES_H
