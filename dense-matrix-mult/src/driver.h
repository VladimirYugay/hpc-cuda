#ifndef DRIVER_HPP
#define DRIVER_HPP

#include "data_types.h"
#include <vector>

float compute(std::vector<float> &C, const std::vector<float> &A, const std::vector<float> &B,
              const Configuration &config);

#endif // DRIVER_HPP
