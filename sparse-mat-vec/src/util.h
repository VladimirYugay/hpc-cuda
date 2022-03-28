#ifndef SPARSE_UTIL_H
#define SPARSE_UTIL_H

#include "data_types.h"
#include <string>
#include <iostream>

std::string modeToStr(ExecutionMode mode);
Configuration makeConfig(int argc, char **argv);
void checkConfig(const Configuration& config);
std::ostream& operator<<(std::ostream& stream, const Configuration& config);
CsrMatrix loadMarketMatrix(const std::string& fileName);

#endif // SPARSE_UTIL_H