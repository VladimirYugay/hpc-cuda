#ifndef STENCIL_AUX_H
#define STENCIL_AUX_H

#include "data_types.h"
#include <iostream>
#include <string>

std::ostream &operator<<(std::ostream &stream, Settings settings);
std::string modeToString(ExecutionMode mode);
size_t getNearestPow2Number(size_t number);

#endif // STENCIL_AUX_H
