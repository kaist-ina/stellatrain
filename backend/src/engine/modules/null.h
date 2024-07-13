#ifndef _ENGINE_MODULES_NULL_
#define _ENGINE_MODULES_NULL_
#include "../module.h"


class ModuleNull : public Module {
private:

public:
    ModuleNull(): Module("Null") {}
    virtual ModuleRunResult run(FasterDpEngine *engine, TrainTaskV2 *task);
    virtual ModuleRunResult run(FasterDpEngine *engine, TrainTaskV2 *task, torch::Tensor tensor);
};


#endif