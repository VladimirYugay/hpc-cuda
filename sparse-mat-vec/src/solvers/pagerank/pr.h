#ifndef SPARSE_PAGERANK_HPP
#define SPARSE_PAGERANK_HPP

#include "data_types.h"

namespace pr {

struct Settings {
    float epsilon{1.0e-4};
    float alpha{0.85};
    size_t numIterations = std::numeric_limits<size_t>::max();
};

double pageRank(CsrMatrix matrix, ExecutionMode mode, Settings settings = Settings());
} // namespace pr

#endif // SPARSE_PAGERANK_HPP
