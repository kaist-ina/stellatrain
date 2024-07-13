#ifndef _ENGINE_MODULES_GRAD_EXCHANGE_
#define _ENGINE_MODULES_GRAD_EXCHANGE_
#include "../module.h"

class ModuleGradExchange : public Module {
private:
    long long int sum_bytes_sent_;
    long long int sum_time_us_;
    double gbps_ewma_;
    int num_items_;
    
public:
    ModuleGradExchange(): Module("GradientExchange"), sum_bytes_sent_(0), sum_time_us_(0), num_items_(0), gbps_ewma_(-1) {}
    virtual ModuleRunResult run(FasterDpEngine *engine, TrainTaskV2 *task);
};

#endif