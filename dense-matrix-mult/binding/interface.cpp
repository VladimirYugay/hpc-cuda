#include "data_types.h"
#include "driver.h"
#include "util.hpp"
#include <iostream>
#include <pybind11/pybind11.h>


void printConfig(Configuration config) {
    std::cout << config;
}

double run(Configuration config) {
    const size_t numElements = config.matrixSize * config.matrixSize;
    std::vector<float> A(numElements, 2.0f);
    std::vector<float> B(numElements, 2.0f);
    std::vector<float> C(numElements, 0.0f);

    return compute(C, A, B, config);
}


namespace py = pybind11;
PYBIND11_MODULE(dense_hpc_aa, module) {
    py::enum_<KernelType>(module, "KernelType")
        .value("KERNEL_CPU", KernelType::KERNEL_CPU)
        .value("KERNEL_GLOBAL", KernelType::KERNEL_GLOBAL)
        .value("KERNEL_TILED", KernelType::KERNEL_TILED)
        .value("KERNEL_COALESCED", KernelType::KERNEL_COALESCED)
        .value("KERNEL_COALESCED_DYM", KernelType::KERNEL_COALESCED_DYM)
        .value("KERNEL_OVERLAPPED", KernelType::KERNEL_OVERLAPPED)
        .value("KERNEL_CUBLAS", KernelType::KERNEL_CUBLAS)
        .export_values();

    py::class_<Configuration>(module, "Configuration")
        .def(py::init<>())
        .def_readwrite("print_matrix", &Configuration::printMatrix)
        .def_readwrite("print_info", &Configuration::printInfo)
        .def_readwrite("tile_size", &Configuration::tileSize)
        .def_readwrite("matrix_size", &Configuration::matrixSize)
        .def_readwrite("num_repeats", &Configuration::numRepeats)
        .def_readwrite("kernel_type", &Configuration::kernelType);

    module.def("print_config", &printConfig, "prints a configuration");
    module.def("run", &run, "runs a configuration");
    module.def("kernel_type_to_str", &kernelTypeToStr, "converts kernel type to string");
    module.def("get_device_name", &getDeviceName, "return device name of the controlled GPU");
}