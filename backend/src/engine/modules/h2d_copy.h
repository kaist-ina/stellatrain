#ifndef _ENGINE_MODULES_H2D_COPY_
#define _ENGINE_MODULES_H2D_COPY_
#include "../module.h"

#include <vector>
#include <c10/cuda/CUDAStream.h>

class ModuleH2DCopyPre : public Module {
public:
    ModuleH2DCopyPre(): Module("H2D_Copy_Pre") {}
    virtual ModuleRunResult run(FasterDpEngine *engine, TrainTaskV2 *task);
};


class ModuleH2DCopy : public Module {
private:
    std::vector<at::cuda::CUDAStream> vec_stream_cpu_to_gpu_copy_;
    
public:
    ModuleH2DCopy(int node_rank): Module("H2D_Copy") {
        for(int i = 0; i < num_workers; i++) {
            vec_stream_cpu_to_gpu_copy_.emplace_back(at::cuda::getStreamFromPool(false, node_rank));
        }
    }
    virtual ModuleRunResult run(FasterDpEngine *engine, TrainTaskV2 *task);
};

class ModuleH2DCopyPost : public Module {
public:
    ModuleH2DCopyPost(): Module("H2D_Copy_Post") {}
    virtual ModuleRunResult run(FasterDpEngine *engine, TrainTaskV2 *task);
};

class FasterDpEngine;
class TrainTask;

#endif