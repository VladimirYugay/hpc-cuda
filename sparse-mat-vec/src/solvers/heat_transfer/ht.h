#ifndef SPARSE_HT_H
#define SPARSE_HT_H

#include "data_types.h"
#include <iostream>
#include <limits>
#include <vector>
#include <mutex>

namespace ht {
struct Settings {
    float conductivity{0.1f};
    float epsilon{1.0e-6};
    size_t numIterations = std::numeric_limits<size_t>::max();
    size_t stepsToPrint{500};
};

enum PlottingStatus { UPDATED, OUTDATED };

struct PoissonSolver {
  explicit PoissonSolver(Settings userSetting = Settings{}) : settings(userSetting) {}

  double run(size_t numGridPoints, ExecutionMode mode);

  std::vector<float> getSolutionVector() {
    std::lock_guard<std::mutex> guard{plottingMutex};
    return x;
  }

  void setPlottingStatus(PlottingStatus newStatus) {
    status = newStatus;
  }

  PlottingStatus getPlottingStatus() {
    return status;
  }

  private:
  void poisson_ellpack();
  void poisson_band();
  void poisson_cusparse();

  Settings settings{};
  std::vector<float> x{};
  std::vector<float> y{};
  size_t numGridPoints{0};

  std::mutex plottingMutex{};
  PlottingStatus status{PlottingStatus::OUTDATED};
};



} // namespace ht

#endif // SPARSE_HT_H
