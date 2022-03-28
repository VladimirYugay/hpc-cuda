#include "solvers/heat_transfer/ht.h"
#include "util.h"
#include "kernels/kernels.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>


namespace py = pybind11;
PYBIND11_MODULE(heat_transfer, module) {
  py::enum_<ExecutionMode>(module, "ExecutionMode")
      .value("HEAT_ELLPACK", ExecutionMode::HEAT_ELLPACK)
      .value("HEAT_CUSPARSE", ExecutionMode::HEAT_CUSPARSE)
      .value("HEAT_BAND", ExecutionMode::HEAT_BAND)
      .export_values();

  py::enum_<ht::PlottingStatus>(module, "PlottingStatus")
      .value("UPDATED", ht::PlottingStatus::UPDATED)
      .value("OUTDATED", ht::PlottingStatus::OUTDATED)
      .export_values();

  py::class_<ht::Settings>(module, "Settings")
      .def(py::init<>())
      .def_readwrite("conductivity", &ht::Settings::conductivity)
      .def_readwrite("epsilon", &ht::Settings::epsilon)
      .def_readwrite("steps_to_print", &ht::Settings::stepsToPrint);

  py::class_<ht::PoissonSolver>(module, "PoissonSolver")
      .def(py::init<ht::Settings>())
      .def("run", &ht::PoissonSolver::run, py::call_guard<py::gil_scoped_release>())
      .def("get_solution_vector", &ht::PoissonSolver::getSolutionVector, py::call_guard<py::gil_scoped_release>())
      .def("set_plotting_status", &ht::PoissonSolver::setPlottingStatus, py::call_guard<py::gil_scoped_release>())
      .def("get_plotting_status", &ht::PoissonSolver::getPlottingStatus, py::call_guard<py::gil_scoped_release>());

  module.def("mode_to_str", &modeToStr, "converts kernel type to string");
  module.def("get_device_name", &getDeviceName, "return device name of the controlled GPU");
}
