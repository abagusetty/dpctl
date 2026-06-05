#include "dpctl4pybind11.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <sycl/sycl.hpp>

// dpctl queues are always in-order, so no sequential order-tracking is needed.
// This module only exposes a helper to submit an empty fence task, used by
// dpctl.SyclTimer's order-manager device timer.
PYBIND11_MODULE(_seq_order_keeper, m)
{
    auto submit_empty_task_fn =
        [](sycl::queue &exec_q,
           const std::vector<sycl::event> &depends) -> sycl::event {
        return exec_q.submit([&](sycl::handler &cgh) {
            cgh.depends_on(depends);
            cgh.single_task([]() {
                // empty body
            });
        });
    };
    m.def("_submit_empty_task", submit_empty_task_fn, py::arg("sycl_queue"),
          py::arg("depends") = py::list());
}
