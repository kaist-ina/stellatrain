#ifndef _ENGINE_MODULE_H
#define _ENGINE_MODULE_H

#include "task.h"
#include <cassert>

#include <torch/torch.h>
#include <torch/extension.h>

class FasterDpEngine;
class TrainTaskV2;

enum ModuleRunResult {
    RESULT_SUCCESS = 0,
    RESULT_FAILURE = 1
};

class Module {
protected:
    const std::string name_;

public:
    Module(std::string name) : name_(name) {}
    inline const std::string &module_name() const { return name_; }

    virtual ModuleRunResult run(FasterDpEngine *engine, TrainTaskV2 *task) {
        assert(false); // not implemented
        return ModuleRunResult::RESULT_SUCCESS;
    }
    virtual ModuleRunResult run(FasterDpEngine *engine, TrainTaskV2 *task, torch::Tensor tensor) {
        assert(false); // not implemented
        return ModuleRunResult::RESULT_SUCCESS;
    }
};

struct CudaCallbackUserData {
    FasterDpEngine *engine;
    TrainTaskV2 *task;
};
#endif