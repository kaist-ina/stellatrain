#ifndef _ENGINE_MODULES_CPU_OPTIMIZE_
#define _ENGINE_MODULES_CPU_OPTIMIZE_
#include "../module.h"

class ModuleCpuOptimize : public Module {
private:

public:
    ModuleCpuOptimize(): Module("CPU_Optimize") {}
    virtual ModuleRunResult run(FasterDpEngine *engine, TrainTaskV2 *task);
};

    size_t unique1d(const torch::Tensor& input, uint32_t* pointer);

#endif