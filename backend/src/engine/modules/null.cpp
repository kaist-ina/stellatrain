#include "null.h"

ModuleRunResult ModuleNull::run(FasterDpEngine *engine, TrainTaskV2 *task) {
    return ModuleRunResult::RESULT_SUCCESS;
}

ModuleRunResult ModuleNull::run(FasterDpEngine *engine, TrainTaskV2 *task, torch::Tensor cpu_tensor) {
    return ModuleRunResult::RESULT_SUCCESS;
}