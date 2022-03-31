#include "solver.h"
#include "kernels/kernels.h"
#include <iostream>
#include <sstream>

using namespace std::chrono;

Solver::Solver(Settings settings) : settings(settings) {
    if ((settings.num1dGridPoints - 2) <= 0) {
        std::runtime_error("field is too small");
    }

    const size_t numThreadsPerBlock = settings.blockSizeX * settings.blockSizeY;
    const size_t maxAllowedNumThreadsPerBlock = getMaxNumThreadPerBlock();
    if (numThreadsPerBlock > maxAllowedNumThreadsPerBlock) {
        std::stringstream stream;
        stream << "requested " << numThreadsPerBlock << " num. threads per block. But "
               << "the device supports only " << maxAllowedNumThreadsPerBlock
               << " threads per block";
        throw std::runtime_error(stream.str());
    }

    fieldSize = (settings.num1dGridPoints - 2) * (settings.num1dGridPoints - 2);
    totalFiledSize = (settings.num1dGridPoints) * (settings.num1dGridPoints);
}


void Solver::init(const std::vector<float> &userField) {
    if (totalFiledSize != userField.size()) {
        throw std::runtime_error("provided field mismatches to the provided solver settings");
    } else {
        dataField = userField;
    }

    solutionField = dataField;
    isInit = true;
}


std::vector<float> Solver::getSolution() {
    std::lock_guard<std::mutex> guard(plottingMutex);
    std::vector<float> currentSolution = solutionField;
    return currentSolution;
}


void Solver::initParams() {
    float dhSquare =
        1.0f / static_cast<float>((settings.num1dGridPoints - 1) * (settings.num1dGridPoints - 1));

    params.dtStable = 0.25f * dhSquare / settings.conductivity;

    params.factor = params.dtStable * settings.conductivity;

    params.invDhSquare = 1.0f / (dhSquare);
}


double Solver::run() {
    if (!isInit) {
        throw std::runtime_error("solver has been not initialized");
    }

    // pre-compute parameters of the solver
    initParams();

    double time{};
    switch (settings.mode) {
        case ExecutionMode::CPU: {
            time = runCpu();
            break;
        }
        case ExecutionMode::GPU_GLOB_MEM: {
            time = runGpu(ExecutionMode::GPU_GLOB_MEM);
            break;
        }
        case ExecutionMode::GPU_SHR_MEM: {
            time = runGpu(ExecutionMode::GPU_SHR_MEM);
            break;
        }
        default: {
            throw std::runtime_error("unknown execution mode");
        }
    }

    return time;
}

void Solver::printField() {
    const auto rowSize = settings.num1dGridPoints;
    for (int j = 0; j < settings.num1dGridPoints; ++j) {
        for (int i = 0; i < settings.num1dGridPoints; ++i) {
            std::cout << dataField[i + j * rowSize] << ' ';
        }
        std::cout << std::endl;
    }
}