#ifndef _ENGINE_MODULES_BARRIER_CHECKER_H_
#define _ENGINE_MODULES_BARRIER_CHECKER_H_
#include "../module.h"

#if DEBUG_BARRIER
class ModuleBarrierChecker : public Module {
private:

public:
    ModuleBarrierChecker(): Module("BarrierChecker") {}
    virtual ModuleRunResult run(FasterDpEngine *engine, TrainTaskV2 *task);
};
#endif

#endif