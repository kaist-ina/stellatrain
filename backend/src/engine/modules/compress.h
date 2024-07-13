#ifndef _ENGINE_MODULES_COMPRESS_
#define _ENGINE_MODULES_COMPRESS_
#include "../module.h"

class ModuleCompress : public Module {
private:

public:
    ModuleCompress(): Module("CPU_Gather") {}
    virtual ModuleRunResult run(FasterDpEngine *engine, TrainTaskV2 *task);
};

#endif