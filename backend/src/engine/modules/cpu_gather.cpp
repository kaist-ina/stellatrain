#include "cpu_gather.h"
#include "../core.h"
#include "../shm_manager.h"
#include "../../misc/array_util.h"
#include <immintrin.h>
#include <memory>
#include <chrono>

#if PROFILER_ENABLED
#include <nvToolsExt.h>
#else
#define nvtxRangePush(x)
#define nvtxRangePop(x)
#endif 

#if DEBUG_ACCURACY
static bool is_first = true;
#endif


static void busy_wait_us(unsigned int microseconds) {
    auto start_time = std::chrono::high_resolution_clock::now();
    auto wait_time = std::chrono::microseconds(microseconds);
    volatile int i = 0;

    while (std::chrono::high_resolution_clock::now() - start_time < wait_time) {
        i++;
        // Busy wait: Do nothing here; just loop until the desired time has elapsed.
    }
}

ModuleRunResult ModuleCpuGather::run(FasterDpEngine *engine, TrainTaskV2 *task) {
    LOG_DEBUG(*this, "Running " + task->key());

    engine->record_stat_end(task, "D2HCopyBarrier");
    engine->record_stat_start(task, "CpuGather");

    const auto local_rank = engine->local_rank();
    const auto num_gpus = engine->node_world_size();

    /**
     * Algorithm:
     * for i in range(n):
     *  start = grad.len * (rank / n)
     *  end = grad.len * ((rank + 1) / n)
     *  grad[0][start:end] += residual[start:end] + grad[1][start:end] + + ... + grad[n-1][start:end]
    */

    auto dst_cpu_grad_tensor = engine->tensor_from_mock(task->shared_cpu_tensor_entry_->grad_[0]);
    
    #if DEBUG_ACCURACY
        float *ptr = reinterpret_cast<float *>(dst_cpu_grad_tensor.data_ptr());
        for (unsigned i = 0; i < dst_cpu_grad_tensor.numel(); i++) {
            assert(!isnan(ptr[i]));
        }
    #endif

    auto tensor_len = dst_cpu_grad_tensor.numel();
    auto start_idx = (tensor_len * local_rank) / num_gpus;
    auto end_idx = (tensor_len * (local_rank + 1)) / num_gpus;
    auto target_len = end_idx - start_idx;

    for (int i = 0; i < num_gpus; i++) {

        float *src;
        float *dst = dst_cpu_grad_tensor.data_ptr<float>() + start_idx;

        if (i == 0) {
            auto &grad_residual_ptr = engine->get_grad_residual(task);
            src = grad_residual_ptr.get() + start_idx;
        } else {
            auto src_cpu_grad_tensor = engine->tensor_from_mock(task->shared_cpu_tensor_entry_->grad_[i]);
            src = src_cpu_grad_tensor.data_ptr<float>() + start_idx;
        }

        engine->record_stat_start(task, "CRIT_PATH_gather_" + std::to_string(i));
#if !(SKIP_SOME_CRITICAL_PATHS)
        add_arrays(dst, src, target_len);
        // busy_wait_us(target_len * 4 / 1000);
#endif
        engine->record_stat_end(task, "CRIT_PATH_gather_" + std::to_string(i));
    }

    task->shared_props_->shared_cpu_data_ready_[local_rank] = true;

    
    engine->record_stat_end(task, "CpuGather");
    engine->record_stat_start(task, "CpuGatherBarrier");
    engine->schedule_after_barrier(engine->module_compress_, task, module_name());
    
    return ModuleRunResult::RESULT_SUCCESS;
}