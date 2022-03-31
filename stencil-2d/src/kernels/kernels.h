#ifndef STENCIL_KERNELS_H
#define STENCIL_KERNELS_H

#include "aux.h"
#include "data_types.h"
#include "kernels.h"
#include <cuda.h>
#include <string>

namespace err {
void checkErr(const std::string &file, int line);
} // namespace err
#define CHECK_ERR err::checkErr(__FILE__, __LINE__)


std::string getDeviceName();
size_t getMaxNumThreadPerBlock();
size_t get1DGrid(size_t blockSize, size_t size);
void launch_computeErr(float *errVector, const float *vector1, const float *vector2, size_t size);


class Reducer {
  public:
    Reducer() = delete;
    explicit Reducer(size_t size);
    ~Reducer();
    Reducer(const Reducer &other) = delete;
    Reducer &operator=(const Reducer &other) = delete;

    float reduceMax(const float *values);

  private:
    ssize_t realVectorSize{};
    ssize_t adjustedSize{};
    float *vectorValues{nullptr};
    float *swapVectorValues{nullptr};
};


void launch_gpuSimple(float *swapField, const float *dataField, const Settings &settings,
                      const Params &params);

void launch_gpuWithShrMem(float *swapField, const float *dataField, const Settings &settings,
                          const Params &params);


#endif // STENCIL_KERNELS_H
