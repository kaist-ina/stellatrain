#ifndef _ENGINE_MODULES_CPU_GATHER_
#define _ENGINE_MODULES_CPU_GATHER_
#include "../module.h"

class ModuleCpuGather : public Module {
private:

public:
    ModuleCpuGather(): Module("CPU_Gather") {}
    virtual ModuleRunResult run(FasterDpEngine *engine, TrainTaskV2 *task);
};

#endif