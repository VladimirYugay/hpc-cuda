#include "data_types.h"
#include "driver.h"
#include "util.hpp"
#include <iostream>
#include <string>
#include <vector>

int main(int argc, char *argv[]) {

    // read the command line and make sure that all provided parameters are Ok
    Configuration config{};
    try {
        config = makeConfig(argc, argv);
        checkConfiguration(config);
    } catch (std::exception &error) {
        std::cout << config;
        std::cout << "ERROR: " << error.what() << std::endl;
        exit(EXIT_FAILURE);
    }

    // print info on the screen
    if (config.printInfo) {
        printDeviceProperties();
    }

    std::cout << std::string(80, '=') << '\n';
    std::cout << config;
    std::cout << std::string(80, '=') << '\n';


    // Allocate and initialise matrices in host memory
    const size_t numElements = config.matrixSize * config.matrixSize;
    std::vector<float> A(numElements, 2.0f);
    std::vector<float> B(numElements, 2.0f);
    std::vector<float> C(numElements, 0.0f);

    // print matrices if required
    if (config.printMatrix) {
        printMatrix("A", A, config.matrixSize);
        printMatrix("B", B, config.matrixSize);
    }

    // NOTE: all compute logic is inside of the compute function
    float spentTime = compute(C, A, B, config);

    // print the resultant matrix if required
    if (config.printMatrix) {
        printMatrix("C", C, config.matrixSize);
    }

    // evaluate performance
    spentTime /= 1000.0f;
    std::cout << "Elapsed time : " << spentTime << " s" << std::endl;

    float operations = (2 * config.matrixSize - 1) * config.matrixSize * config.matrixSize;
    operations *= config.numRepeats;
    std::cout << "operations: " << operations << std::endl;

    constexpr float FACTOR = 1024;
    const float GFLOPS = (static_cast<float>(operations) / spentTime) / (FACTOR * FACTOR * FACTOR);
    std::cout << "Performance: " << GFLOPS << " GFlop/s" << std::endl;

    return 0;
}