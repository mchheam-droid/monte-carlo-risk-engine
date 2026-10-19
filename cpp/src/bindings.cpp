#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "simulation_engine.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_risk_engine, m) {
    m.doc() = "High-Performance Monte Carlo Portfolio Risk Engine (C++)";

    //  Bind AssetData 
    py::class_<risk::AssetData>(m, "AssetData")
        .def(py::init<double, double, double, double>(),
             py::arg("spot"), py::arg("weight"), py::arg("vol"), py::arg("drift"))
        .def_readwrite("spot", &risk::AssetData::spot)
        .def_readwrite("weight", &risk::AssetData::weight)
        .def_readwrite("vol", &risk::AssetData::vol)
        .def_readwrite("drift", &risk::AssetData::drift);

    // Bind SimulationEngine 
    py::class_<risk::SimulationEngine>(m, "SimulationEngine")
        .def(py::init<int, uint64_t>(),
             py::arg("num_threads") = 0,
             py::arg("seed") = 42)
        .def("simulate_portfolio_values", 
             &risk::SimulationEngine::simulate_portfolio_values,
             py::arg("assets"),
             py::arg("correlation_matrix"),
             py::arg("num_paths"),
             py::arg("horizon_days") = 1.0,
             py::arg("use_antithetic") = true,
             py::call_guard<py::gil_scoped_release>(), 
             "Simulates terminal portfolio values using correlated Geometric Brownian Motion.");
}