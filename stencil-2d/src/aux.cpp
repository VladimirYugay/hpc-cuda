#include "aux.h"
#include <cmath>
#include <map>


std::ostream &operator<<(std::ostream &stream, Settings settings) {
    stream << "num. grid points along x and y: " << settings.num1dGridPoints << '\n'
           << "total num. grid points: " << settings.num1dGridPoints * settings.num1dGridPoints
           << '\n'
           << "num. iterations: " << settings.numIterations << '\n'
           << "steps per print: " << settings.stepsPerPrint << '\n'
           << "eps: " << settings.eps << '\n'
           << "end time: " << settings.endTime << '\n'
           << "mode: " << modeToString(settings.mode);

    return stream;
}


std::string modeToString(ExecutionMode mode) {
    static std::map<ExecutionMode, std::string> map{
        {ExecutionMode::CPU, "cpu"},
        {ExecutionMode::GPU_GLOB_MEM, "gpu-global-memory"},
        {ExecutionMode::GPU_SHR_MEM, "gpu-shared-memory"}};

    if (map.find(mode) == map.end()) {
        throw std::runtime_error("unknown execution mode");
    }
    return map[mode];
}


size_t getNearestPow2Number(size_t number) {
    auto power = std::ceil(std::log2(static_cast<double>(number)));
    return std::pow(2, power);
}