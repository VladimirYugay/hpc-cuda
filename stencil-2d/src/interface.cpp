#include "aux.h"
#include "data_types.h"
#include "solver.h"
#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>

void printSettings(const Settings &settings) {
    std::cout << std::string(80, '=') << '\n';
    std::cout << settings << '\n';
    std::cout << std::string(80, '=') << '\n';
}

namespace py = pybind11;

PYBIND11_MODULE(interface, module) {
    py::enum_<ExecutionMode>(module, "ExecutionMode")
        .value("CPU", ExecutionMode::CPU)
        .value("GPU_GLOB_MEM", ExecutionMode::GPU_GLOB_MEM)
        .value("GPU_SHR_MEM", ExecutionMode::GPU_SHR_MEM)
        .export_values();

    py::class_<Settings>(module, "Configuration")
        .def(py::init<>())
        .def_readwrite("num_1d_grid_points", &Settings::num1dGridPoints)
        .def_readwrite("num_iterations", &Settings::numIterations)
        .def_readwrite("steps_per_print", &Settings::stepsPerPrint)
        .def_readwrite("conductivity", &Settings::conductivity)
        .def_readwrite("mode", &Settings::mode)
        .def_readwrite("block_size_x", &Settings::blockSizeX)
        .def_readwrite("block_size_y", &Settings::blockSizeY)
        .def_readwrite("eps", &Settings::eps)
        .def_readwrite("end_time", &Settings::endTime);

    py::class_<Solver>(module, "Solver")
        .def(py::init<Settings>())
        .def("run", &Solver::run, py::call_guard<py::gil_scoped_release>())
        .def("init", &Solver::init, py::call_guard<py::gil_scoped_release>())
        .def("get_solution", &Solver::getSolution, py::call_guard<py::gil_scoped_release>())
        .def("print_field", &Solver::printField, py::call_guard<py::gil_scoped_release>());

    module.def("print_config", &printSettings, "prints config");
}
