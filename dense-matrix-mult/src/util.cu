#include "util.hpp"
#include <boost/program_options.hpp>
#include <iostream>
#include <set>
#include <sstream>
#include <unordered_map>

namespace po = boost::program_options;

namespace err {
std::string PrevFile{};
int PrevLine{0};

/**
 * helper function to check for errors in CUDA calls
 * source: NVIDIA
 * */
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
} // namespace err


void printMatrix(const std::string &name, const std::vector<float> &matrix, int size) {
    std::cout << "Matrix " << name << " (size: " << size << ")\n";
    for (int column = 0; column < size; ++column) {
        for (int row = 0; row < size; ++row) {
            std::cout << matrix[row * size + column] << ' ';
        }
        std::cout << '\n';
    }
    std::cout << std::endl;
}


/* initialise Matrix: discrete Sine Transform */
void dstMatrix(std::vector<float> &A, const int size) {
    constexpr float PI = 3.14159265;
    for (int i = 0; i < size; ++i) {
        for (int k = 0; k < size; ++k) {
            A[i * size + k] = std::sin(((i + 1) * (k + 1) * PI) / (size + 1));
        }
    }
}

std::vector<float> transpose(const std::vector<float> &matrix, const size_t size) {
    std::vector<float> transposedMatrx(matrix.size(), 0.0);
    const size_t numRows = matrix.size() / size;
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            transposedMatrx[i + size * j] = matrix[j + size * i];
        }
    }
    return transposedMatrx;
}

// Print device info
void printDeviceInfo(const cudaDeviceProp &devProp) {
    printf("Revision number:               %d.%d\n", devProp.major, devProp.minor);
    printf("Name:                          %s\n", devProp.name);
    printf("Total global memory:           %lu MB\n", devProp.totalGlobalMem / (1024 * 1024));
    printf("Total shared memory per block: %lu kB\n", devProp.sharedMemPerBlock / 1024);
    printf("Total registers per block:     %d\n", devProp.regsPerBlock);
    printf("Warp size:                     %d\n", devProp.warpSize);
    printf("Maximum memory pitch:          %lu MB\n", devProp.memPitch / (1024 * 1024));
    printf("Maximum threads per block:     %d\n", devProp.maxThreadsPerBlock);

    printf("Maximum dimensions of block:   %d %d %d\n", devProp.maxThreadsDim[0],
           devProp.maxThreadsDim[1], devProp.maxThreadsDim[2]);
    printf("Maximum dimensions of grid:    %d %d %d\n", devProp.maxGridSize[0],
           devProp.maxGridSize[1], devProp.maxGridSize[2]);

    printf("Clock rate:                    %d MHz\n", devProp.clockRate / 1000);
    printf("Total constant memory:         %lu kB\n", devProp.totalConstMem / 1024);
    printf("Texture alignment:             %lu B\n", devProp.textureAlignment);
    printf("Concurrent copy and execution: %s\n", (devProp.deviceOverlap ? "Yes" : "No"));
    printf("Number of multiprocessors:     %d\n", devProp.multiProcessorCount);
    printf("Kernel execution timeout:      %s\n",
           (devProp.kernelExecTimeoutEnabled ? "Yes" : "No"));
    printf("\n");
}


void printDeviceProperties() {
    int devCount{};
    cudaGetDeviceCount(&devCount);
    CHECK_ERR;
    std::cout << "CUDA Device Query...\n";
    std::cout << "====================== CUDA DEVICE QUERY ==============================="
              << std::endl;
    std::cout << "There are " << devCount << " CUDA devices.\n";

    // Iterate through devices
    for (int i = 0; i < devCount; ++i) {
        // Get device properties
        printf("====================== CUDA device #%d===================================\n", i);
        cudaDeviceProp devProp{};
        cudaGetDeviceProperties(&devProp, i);
        CHECK_ERR;
        printDeviceInfo(devProp);
    }
}


std::string getDeviceName() {
    int deviceId{-1};
    cudaGetDevice(&deviceId);

    cudaDeviceProp devProp{};
    cudaGetDeviceProperties(&devProp, deviceId);
    std::stringstream stream;

    stream << devProp.name << ", Compute Capability: " << devProp.major << '.' << devProp.minor;
    return stream.str();
}

std::string kernelTypeToStr(KernelType type) {
    static std::map<KernelType, std::string> map{
        {KernelType::KERNEL_CPU, "cpu"},
        {KernelType::KERNEL_GLOBAL, "global"},
        {KernelType::KERNEL_TILED, "tiled"},
        {KernelType::KERNEL_COALESCED, "coalesced"},
        {KernelType::KERNEL_COALESCED_DYM, "coalesced_dym"},
        {KernelType::KERNEL_OVERLAPPED, "overlapped"},
        {KernelType::KERNEL_CUBLAS, "cublas"}};
    return map[type];
}


std::ostream &operator<<(std::ostream &stream, const Configuration &config) {
    stream << "Print Matrix: " << std::boolalpha << config.printMatrix << '\n'
           << "Print Info: " << config.printInfo << '\n'
           << "Tile Size: " << config.tileSize << '\n'
           << "Matrix Size: " << config.matrixSize << '\n'
           << "Num. Repeats: " << config.numRepeats << '\n'
           << "Kernel Type: " << kernelTypeToStr(config.kernelType) << '\n';
    return stream;
}


void checkConfiguration(const Configuration &config) {
    std::set<int> allowedTileSize{4, 8, 16, 32};
    if (allowedTileSize.find(config.tileSize) == allowedTileSize.end()) {
        std::stringstream stream;
        stream << "Allowed tile sizes: ";
        for (auto item : allowedTileSize) {
            stream << item << ", ";
        }
        stream << '\n';
        throw std::runtime_error(stream.str());
    }

    if (config.matrixSize <= 0) {
        throw std::runtime_error("matrix size must be positive");
    }

    if (config.tileSize > config.matrixSize) {
        throw std::runtime_error("Matrix size must be bigger than the tile size");
    }

    if (config.numRepeats <= 0) {
        throw std::runtime_error("Num repeats must be a positive integer number");
    }

    if (config.matrixSize % config.tileSize != 0) {
        throw std::runtime_error("Error: matrix size has to be a multiple of tile size %d \n");
    };
}


Configuration makeConfig(int argc, char **argv) {
    Configuration config;

    po::options_description description("HPC-AA Matrix Multiplication Exercise.");
    std::string kernelStr{};
    // clang-format off
    description.add_options()
        ("help,h", "produce help message")
        ("print-matrix,m", po::value<bool>(&config.printMatrix)->default_value(false), "print the input and result matrices")
        ("print-device-info,d", po::value<bool>(&config.printInfo)->default_value(false), "print information about the available GPU devices")
        ("tile-size,t", po::value<int>(&config.tileSize)->default_value(8), "Tile size for the Matrix Multiplication. Must be either 4, 8, 16, 32 [Default: 4]")
        ("matrix-size,n", po::value<int>(&config.matrixSize)->default_value(8), "Size n of the matrix. Final size: n*n floats")
        ("repeats,r", po::value<int>(&config.numRepeats)->default_value(1), "Number of times the matrix multiplication is repeated")
        ("kernel-type,k", po::value<std::string>(&kernelStr)->default_value("cpu"), "Type of the kernel to be executed. Must be one of: cpu, global, tiled, coalesced, coalesced_dym, overlapped, cublas")
    ;
    // clang-format on

    po::variables_map cmdMap;
    po::store(po::parse_command_line(argc, argv, description), cmdMap);
    po::notify(cmdMap);

    if (cmdMap.count("help")) {
        std::cout << description << std::endl;
        exit(EXIT_SUCCESS);
    }
    std::unordered_map<std::string, KernelType> strToType{
        {"cpu", KernelType::KERNEL_CPU},
        {"global", KernelType::KERNEL_GLOBAL},
        {"tiled", KernelType::KERNEL_TILED},
        {"coalesced", KernelType::KERNEL_COALESCED},
        {"coalesced_dym", KernelType::KERNEL_COALESCED_DYM},
        {"overlapped", KernelType::KERNEL_OVERLAPPED},
        {"cublas", KernelType::KERNEL_CUBLAS}};

    if (strToType.find(kernelStr) == strToType.end()) {
        std::cout << "unknown kernel type: " << kernelStr << '\n';
        std::cout << "allowed options: ";
        for (const auto &item : strToType) {
            std::cout << item.first << ", ";
        }
        std::cout << std::endl;
        exit(EXIT_FAILURE);
    } else {
        config.kernelType = strToType[kernelStr];
    }

    return config;
}