#ifndef SPARSE_AUX_HPP
#define SPARSE_AUX_HPP

#include "data_types.h"

void convertCooToCsr(CsrMatrix &matrix, const std::vector<int>& I);
EllMatrix get1DStencilEllMatrix(int numPoints);
CsrMatrix get1DStencilCsrMatrix(int numPoints);


#endif // SPARSE_AUX_HPP
