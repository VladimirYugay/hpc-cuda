#ifndef STENCIL_SOLVER_H
#define STENCIL_SOLVER_H

#include "data_types.h"
#include <iostream>
#include <mutex>
#include <vector>

class Solver {
  public:
    Solver() = delete;
    explicit Solver(Settings settings);
    Solver(const Solver &other) = default;
    Solver &operator=(const Solver &other) = default;
    ~Solver() = default;

    void init(const std::vector<float> &userField);
    std::vector<float> getSolution();
    double run();
    void printField();
    void setExecutionMode(ExecutionMode userMode) {
        settings.mode = userMode;
    }

  private:
    void initParams();
    double runCpu();
    double runGpu(ExecutionMode mode);

    Settings settings;
    Params params;
    std::vector<float> dataField{};
    std::vector<float> solutionField{};
    std::mutex plottingMutex;

    size_t fieldSize{};
    size_t totalFiledSize{};
    bool isInit{false};
};

#endif // STENCIL_SOLVER_H
