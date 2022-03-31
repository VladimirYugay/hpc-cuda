#include "data_types.h"
#include "solver.h"
#include <iostream>
#include <vector>


int main(int argc, char *argv[]) {
    Settings settings;
    settings.num1dGridPoints = 10000;
    settings.numIterations = 100;
    settings.stepsPerPrint = 100;
    settings.blockSizeX = 32;
    settings.blockSizeY = 16;
    settings.eps = 0.0;
    settings.endTime = std::numeric_limits<float>::max();
    settings.isVerbose = false;

    Solver solver(settings);

    const size_t size = settings.num1dGridPoints * settings.num1dGridPoints;
    std::vector<float> field(size, 1.0f);
    solver.init(field);

    solver.setExecutionMode(ExecutionMode::GPU_GLOB_MEM);
    auto time = solver.run();
    std::cout << "GPU_GLOB_MEM time,s : " << time << std::endl;

    solver.setExecutionMode(ExecutionMode::GPU_SHR_MEM);
    time = solver.run();
    std::cout << "GPU_SHR_MEM time,s : " << time << std::endl;

    return 0;
}