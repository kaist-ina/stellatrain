#include "grad_exchange.h"
#include "../core.h"
#include "../shm_manager.h"
#include "../comm_manager.h"
#include <zmq_addon.hpp>

#include <string>
#include <chrono>

#if PROFILER_ENABLED
#include <nvToolsExt.h>
#else
#define nvtxRangePush(x)
#define nvtxRangePop(x)
#endif 


ModuleRunResult ModuleGradExchange::run(FasterDpEngine *engine, TrainTaskV2 *task) {
    LOG_DEBUG(*this, "Running " + task->key());
    
    if (!engine->is_node_master()) {
        engine->schedule_after_barrier(engine->module_h2d_copy_pre_, task, module_name());
        return ModuleRunResult::RESULT_SUCCESS;
    }
    
    engine->record_stat_start(task, "GradExchange");

    // const std::string key(task->persistent_key_);
    // std::cerr << "ModuleGradExchange " << task->key_ << std::endl;

    // this will only be run by one process per node
    // no need to worry synchronization

    // here we assume two things:
    // 1. the number of compressed element is same for all nodes
    // 2. we have pre-allocated buffer for task->shared_grad_val_, size of  task->tensor_compressed_numel_ * WORLD_SIZE

    auto &mgr = engine->comm_manager();
    assert(task->grad_sync_iter_ < engine->world_size());


    // zeromq push-pull socket does not ensure in-order delivery!
    // this means that second iteration may arrive before first iteration

    if (task->grad_sync_iter_ + 1 < engine->world_size()) {
        
        const auto numel = task->tensor_compressed_numel_;
        auto ptr_grad_snd_idx = task->compressed_grad_idx_ptr_ + task->grad_sync_iter_ * numel;
        auto ptr_grad_snd_val = task->compressed_grad_val_ptr_ + task->grad_sync_iter_ * numel;

        auto ptr_grad_rcv_idx = task->compressed_grad_idx_ptr_ + (task->grad_sync_iter_ + 1 ) * numel;
        auto ptr_grad_rcv_val = task->compressed_grad_val_ptr_ + (task->grad_sync_iter_ + 1 ) * numel;

        auto iter = task->grad_sync_iter_; // need this as we will increment grad_sync_iter_ below
        task->grad_sync_iter_++;
        if (iter == 0)
            task->grad_sync_start_ = std::chrono::steady_clock::now();
        mgr->queueTx(task, ptr_grad_snd_idx, ptr_grad_snd_val, iter);
        engine->schedule_after_comm(engine->module_grad_exchange_, task, ptr_grad_rcv_val, ptr_grad_rcv_idx, numel, iter);
        // enqueue transfer and reschedule this task.
        return ModuleRunResult::RESULT_SUCCESS;
        
    } else {
        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        auto time_us = std::chrono::duration_cast<std::chrono::microseconds>(end - task->grad_sync_start_).count();
        auto bytes_sent = task->tensor_compressed_numel_ * (engine->world_size() - 1) * sizeof(float);
        
        sum_bytes_sent_ += bytes_sent;
        sum_time_us_ += time_us;
        num_items_++;

        task->tensor_compressed_numel_ *= engine->world_size();
        
        engine->record_stat_end(task, "GradExchange");
        engine->schedule(engine->module_cpu_optimize_, task);
        return ModuleRunResult::RESULT_SUCCESS;
    }

    assert(false); // raise
    return ModuleRunResult::RESULT_FAILURE;
}