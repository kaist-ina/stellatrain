#ifndef _ENGINE_MODULES_D2H_COPY_
#define _ENGINE_MODULES_D2H_COPY_
#include "../module.h"

#include <vector>
#include <c10/cuda/CUDAStream.h>

class ModuleD2HCopy : public Module {
private:
    std::vector<at::cuda::CUDAStream> vec_stream_gpu_to_cpu_copy_;

public:
    ModuleD2HCopy(int node_rank): Module("D2H_Copy") {
        for(int i = 0; i < num_workers; i++) {
            vec_stream_gpu_to_cpu_copy_.emplace_back(at::cuda::getStreamFromPool(false, node_rank));
        }
    }
    virtual ModuleRunResult run(FasterDpEngine *engine, TrainTaskV2 *task);
};

class ModuleD2HCopyPost : public Module {
public:
    ModuleD2HCopyPost(): Module("D2H_Copy_Post") {}
    virtual ModuleRunResult run(FasterDpEngine *engine, TrainTaskV2 *task);
};

#endif