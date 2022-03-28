#include "ht.h"
#include <chrono>
#include <iostream>
#include <vector>

using namespace std::chrono;

namespace ht {
double PoissonSolver::run(size_t numGridPoints, ExecutionMode mode) {

    // set initial state
    this->numGridPoints = numGridPoints;
    x.resize(this->numGridPoints, 1.0f);
    y.resize(this->numGridPoints, 0.0f);

    high_resolution_clock::time_point start = high_resolution_clock::now();
    switch (mode) {
        case ExecutionMode::HEAT_CUSPARSE: {
            poisson_cusparse();
            break;
        }
        case ExecutionMode::HEAT_ELLPACK: {
            poisson_ellpack();
            break;
        }
        case ExecutionMode::HEAT_BAND: {
            poisson_band();
            break;
        }
        default: {
            throw std::runtime_error("Execution aborted: Unknown execution mode for the heat transfer solver");
        }
    }

    high_resolution_clock::time_point end = high_resolution_clock::now();
    return duration_cast<duration<double>>(end - start).count();
}
} // namespace ht