#include <functional>

#include <pybind11/pybind11.h>
#include <pybind11/embed.h>
#include <pybind11/stl.h>

#include <torch/torch.h>
#include <torch/extension.h>
#include <vector>

#include "../optim/sgd.h"
#include "../optim/adam.h"
#include "../engine/core.h"

namespace py = pybind11;


PYBIND11_MODULE(fasterdp, m) {
    m.doc() = "pybind11 FasterDP implementation"; // optional module docstring
    m.attr("__version__") = "v1.0.0";

    py::class_<SGD>(m, "SGD")
        .def(py::init<>())
        .def_property("lr", &SGD::get_lr, &SGD::set_lr)
        .def_property("momentum", &SGD::get_momentum, &SGD::set_momentum)
        .def_property("smart_momentum", &SGD::get_smart_momentum, &SGD::set_smart_momentum)
        .def("optimize", &SGD::optimize);
    
    py::class_<Adam>(m, "Adam")
        .def(py::init<>())
        .def_property("lr", &Adam::get_lr, &Adam::set_lr)
        .def_property("b1", &Adam::get_b1, &Adam::set_b1)
        .def_property("b2", &Adam::get_b2, &Adam::set_b2)
        .def_property("weight_decay", &Adam::get_weight_decay, &Adam::set_weight_decay)
        .def_property("eps", &Adam::get_eps, &Adam::set_eps)
        .def("optimize", &Adam::optimize);

    m.def("gather", [](std::vector<torch::Tensor> lst) {
        // will do nothing
    });

    m.def("configure_compression", [](const std::string method) {
        FasterDpEngine::getInstance().configure_compression(method);
    });

    m.def("configure", [] (
        std::string master_addr, uint16_t master_port, int world_size, int rank, int local_session_id, 
        int local_world_size=0, int local_rank=0, const std::string method="", int gradient_accumulation=1) {
        FasterDpEngine::getInstance().configure(master_addr, master_port, world_size, rank, local_session_id, local_world_size, local_rank, method, gradient_accumulation);
    });

    m.def("gradient_accumulation", [] () {
        return FasterDpEngine::getInstance().gradient_accumulation();   
    });

    m.def("is_debug_accuracy_mode", [] () {
        return FasterDpEngine::getInstance().is_debug_accuracy_mode();   
    });

    m.def("barrier", []() {
        FasterDpEngine::getInstance().barrier();
    });

    m.def("pre_train_init", [](int layer_idx, std::string &name, torch::Tensor tensor) {
        return FasterDpEngine::getInstance().pre_train_init(layer_idx, name, tensor);
    });

    m.def("post_backward_process", [](int layer_idx, std::string &name, torch::Tensor tensor, torch::Tensor param_tensor) {
        return FasterDpEngine::getInstance().post_backward_process(layer_idx, name, tensor, param_tensor);
    });

    m.def("pre_forward_process", [](int layer_idx, std::string &name) {
        return FasterDpEngine::getInstance().pre_forward_process(layer_idx, name);
    });

    m.def("synchronize", []() {
        FasterDpEngine::getInstance().synchronize_backend();
    });

    m.def("compress", [](std::string &name, torch::Tensor tensor, float ratio) {
        return FasterDpEngine::getInstance().compress(name, tensor, ratio);
    });

    m.def("set_optimizer", [](std::string &optimizer_name) {
        return FasterDpEngine::getInstance().get_sparse_optimizer(optimizer_name);
    });

    m.def("configure_optimizer", [](std::string &option_name, float val) {
        return FasterDpEngine::getInstance().sparse_optimizer()->configure(option_name, val);
    });

    m.def("configure_optimizer", [](std::string &option_name, bool val) {
        return FasterDpEngine::getInstance().sparse_optimizer()->configure(option_name, val);
    });
    
    m.def("configure_compression_ratio", [](double ratio) {
        return FasterDpEngine::getInstance().configure_compression_ratio(ratio);
    });

    m.def("force_model_sync", [](int layer_idx, std::string &name, bool dry_run = false) {
        return FasterDpEngine::getInstance().force_model_sync(layer_idx, name, dry_run);
    });
}