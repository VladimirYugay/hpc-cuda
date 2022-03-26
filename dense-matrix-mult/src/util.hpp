#ifndef UTIL_HPP
#define UTIL_HPP

#include "data_types.h"
#include <string>
#include <vector>

namespace err {
void checkErr(const std::string &file, int line);
}
#define CHECK_ERR err::checkErr(__FILE__, __LINE__)

void printMatrix(const std::string &name, const std::vector<float> &matrix, int size);
std::vector<float> transpose(const std::vector<float> &matrix, const size_t size);

template <typename T>
std::ostream &operator<<(std::ostream &stream, const std::vector<T> &container) {
    for (size_t i = 0; i < container.size() - 1; ++i) {
        stream << container[i] << ", ";
    }
    stream << container[container.size() - 1];
    return stream;
}

void printDeviceProperties();
std::string getDeviceName();
std::string kernelTypeToStr(KernelType type);
std::ostream &operator<<(std::ostream &stream, const Configuration &config);
void checkConfiguration(const Configuration &config);
Configuration makeConfig(int argc, char **argv);

#endif // UTIL_HPP
