#include "config.h"
#include "core_internal.h"
#include "modules/d2h_copy.h"
#include "modules/cpu_gather.h"
#include "modules/compress.h"
#include "modules/null.h"
#include "modules/grad_exchange.h"
#include "modules/cpu_optimize.h"
#include "modules/h2d_copy.h"
#include "modules/barrier_checker.h"

#include "../optim/sgd.h"
#include "../optim/adam.h"

void FasterDpEngine::load_modules() {
    module_d2h_copy_ = std::make_unique<ModuleD2HCopy>(local_rank_);
    module_d2h_copy_post_ = std::make_unique<ModuleD2HCopyPost>();
    module_cpu_gather_ = std::make_unique<ModuleCpuGather>();
    module_compress_ = std::make_unique<ModuleCompress>();
    module_grad_exchange_ = std::make_unique<ModuleGradExchange>();
    module_cpu_optimize_ = std::make_unique<ModuleCpuOptimize>();
    module_h2d_copy_pre_ = std::make_unique<ModuleH2DCopyPre>();
    module_h2d_copy_ = std::make_unique<ModuleH2DCopy>(local_rank_);
    module_h2d_copy_post_ = std::make_unique<ModuleH2DCopyPost>();
#if DEBUG_BARRIER
    module_barrier_checker_ = std::make_unique<ModuleBarrierChecker>();
#endif
}