#include "compress.h"
#include "../core.h"
#include "../shm_manager.h"
#include <immintrin.h>
#include <string>
#include <iostream>

#if PROFILER_ENABLED
#include <nvToolsExt.h>
#else
#define nvtxRangePush(x)
#define nvtxRangePop(x)
#endif 

#if DEBUG_ACCURACY
static bool is_first = true;
#endif

ModuleRunResult ModuleCompress::run(FasterDpEngine *engine, TrainTaskV2 *task) {
    LOG_DEBUG(*this, "Running " + task->key());
    
    engine->record_stat_end(task, "CpuGatherBarrier");

    if (!engine->is_node_master()) {
        if (engine->world_size() == 1) {
            engine->record_stat_start(task, "CpuOptimizeBarrier");
            engine->schedule_after_barrier(engine->module_h2d_copy_pre_, task, "CPU_Optimize");
        } else {
            engine->schedule(engine->module_grad_exchange_, task);
        }
        return ModuleRunResult::RESULT_SUCCESS;
    }
    // this will only be run by one process per node
    // no need to worry synchronization

    engine->record_stat_start(task, "Compress");

#if MERGE
    // TODO SIMD alignment
    int64_t piece_start = 0;
    int64_t piece_end = task->tensor_numel_ - (task->tensor_numel_ % 4);
    const int64_t piece_len = piece_end - piece_start;

    const float k = (1 - engine-> compression_ratio()) / static_cast<float>(engine->world_size());

    float *gather_grad_cpu_data_ptr = engine->tensor_from_mock(task->shared_cpu_tensor_entry_->grad_[0]).data_ptr<float>();

    auto src = std::make_pair(gather_grad_cpu_data_ptr, task->tensor_numel_);

    assert(task->tensor_compressed_numel_ == 0);

    const int64_t original_numel = std::max(static_cast<int64_t>(std::min(static_cast<int64_t>(task->tensor_numel_), 1L)), static_cast<int64_t>(task->tensor_numel_ * k));
    const int64_t numel = original_numel * engine->world_size();

    task->tensor_compressed_numel_ = original_numel;
    // allocate
    assert(task->compressed_grad_val_ptr_ == nullptr);
    assert(task->compressed_grad_idx_ptr_ == nullptr);
    
    task->compressed_grad_val_ptr_ = new float[numel];
    task->compressed_grad_idx_ptr_ = new uint32_t[numel];

    memset(task->compressed_grad_val_ptr_, 0, sizeof(float) * numel);
    memset(task->compressed_grad_idx_ptr_, 0, sizeof(uint32_t) * numel);

    auto seg_idx = std::make_pair(task->compressed_grad_idx_ptr_ , task->tensor_compressed_numel_);
        
    auto seg_val = std::make_pair(task->compressed_grad_val_ptr_, task->tensor_compressed_numel_);
    
    uint32_t expected_compressed_grad_elems = k * task->tensor_numel_;

#else
    
    const int64_t piece_count = engine->world_size();
    const int64_t piece_id = (task->iter() + engine->total_rank()) % piece_count;
    int64_t piece_start = piece_id * task->tensor_numel_ / piece_count;
    int64_t piece_end = (piece_id + 1) * task->tensor_numel_ / piece_count;

    // for SIMD alignment
    piece_start = piece_start - (piece_start % 4); // round down
    piece_end = piece_end - (piece_end % 4); // round down also
    if (piece_id + 1 == piece_count)
        piece_end = task->tensor_numel_;
    const int64_t piece_len = piece_end - piece_start;
    
    const float original_k = (1 - engine->compression_ratio());
    const float k = original_k / piece_count;

    // std::cerr << "Rank=" << engine->total_rank() << ", iter=" << task->iter() << ", Selecting [" << piece_start << ", " << piece_end << ")" << std::endl;
    assert(piece_len > 0);

    float *gather_grad_cpu_data_ptr = engine->tensor_from_mock(task->shared_cpu_tensor_entry_->grad_[0]).data_ptr<float>();
    auto original_src = std::make_pair(gather_grad_cpu_data_ptr, task->tensor_numel_);
    auto src = std::make_pair(gather_grad_cpu_data_ptr + piece_start, piece_len);

    /** Residual is already added to gradient tensor in CPU_Gather Step.*/

    assert(task->tensor_compressed_numel_ == 0);

    const int64_t numel = std::max(static_cast<int64_t>(std::min(static_cast<int64_t>(task->tensor_numel_), 1L)), static_cast<int64_t>(task->tensor_numel_ * k));
    const int64_t original_numel =  numel * piece_count; //std::max(static_cast<int64_t>(std::min(static_cast<int64_t>(task->tensor_numel_), 1L)), static_cast<int64_t>(task->tensor_numel_ * original_k));

    task->tensor_compressed_numel_ = numel;
    // allocate
    assert(task->compressed_grad_val_ptr_ == nullptr);
    assert(task->compressed_grad_idx_ptr_ == nullptr);
    
    task->compressed_grad_val_ptr_ = new float[original_numel];
    task->compressed_grad_idx_ptr_ = new uint32_t[original_numel];

    memset(task->compressed_grad_val_ptr_, 0, sizeof(float) * original_numel);
    memset(task->compressed_grad_idx_ptr_, 0, sizeof(uint32_t) * original_numel);

    auto seg_idx = std::make_pair(task->compressed_grad_idx_ptr_ , task->tensor_compressed_numel_);
        
    auto seg_val = std::make_pair(task->compressed_grad_val_ptr_, task->tensor_compressed_numel_);
    
    uint32_t expected_compressed_grad_elems = k * task->tensor_numel_;

#endif

#if DEBUG_ACCURACY
    assert(engine->compressor());
    if (is_first) {
        is_first = false;
        std::cerr << "Using Compressor: " << engine->compressor()->name() << std::endl;
    }


    assert (task->tensor_numel_ >= expected_compressed_grad_elems);
    assert (task->tensor_numel_ >= task->tensor_compressed_numel_);
   
    for (unsigned i = 0 ;i < original_src.second; i++) {
        assert(!isnan(original_src.first[i]));
    }
    float norm_before = torch::from_blob(src.first, src.second).norm().item<float>();
#endif

#if MERGE
    engine->record_stat_start(task, "CRIT_PATH_compress");
    engine->compressor()->compress(task->persistent_key(), src, original_numel, seg_idx, seg_val, 0);
    engine->record_stat_end(task, "CRIT_PATH_compress");
#else
    engine->record_stat_start(task, "CRIT_PATH_compress");
    engine->compressor()->compress(task->persistent_key(), src, numel, seg_idx, seg_val, piece_start);
    engine->record_stat_end(task, "CRIT_PATH_compress");
#endif

#if DEBUG_ACCURACY
    float norm_after = torch::from_blob(src.first, src.second).norm().item<float>();
    if (norm_before != norm_after) {
        std::stringstream s;
        s << "[" << task->iter() << "] " << task->persistent_key() << ": Data corrupt during compression. Expected norm = " << norm_before << ", but got " << norm_after << ". at addresss = " << src.first;
        std::cerr << s.str() << std::endl;
        assert(norm_before == norm_after);
    }


    for (unsigned i = 0 ; i < task->tensor_compressed_numel_; i++) {
        assert (task->compressed_grad_idx_ptr_[i] < task->tensor_numel_);
        
        if (task->compressed_grad_val_ptr_[i] != original_src.first[task->compressed_grad_idx_ptr_[i]] && task->compressed_grad_val_ptr_[i] != 0) {
            std::stringstream s;
            s << "Invalid compressed gradients. Expected " << original_src.first[task->compressed_grad_idx_ptr_[i]] << " (at index " << task->compressed_grad_idx_ptr_[i] << "), but got ";
            s << "compressed[" << i << "] = (" << task->compressed_grad_idx_ptr_[i] << " -> " << task->compressed_grad_val_ptr_[i] << ").";
            std::cerr << s.str() << std::endl;
            assert (task->compressed_grad_val_ptr_[i] == original_src.first[task->compressed_grad_idx_ptr_[i]]);
        }
    }
#endif

#if MERGE
    // Compensation - save
    {
        auto &grad_residual_ptr = engine->get_grad_residual(task);
        
        // zero out selected gradients
        for (std::size_t i = 0; i < seg_idx.second; i++)
            src.first[seg_idx.first[i]] = 0;

        // save to residual
        /* TODO: Improve here2 */
        engine->record_stat_start(task, "CRIT_PATH_save_residual");
#if !(SKIP_SOME_CRITICAL_PATHS)
        memcpy(grad_residual_ptr.get(), src.first, sizeof(float) * task->tensor_numel_);
#endif
        engine->record_stat_end(task, "CRIT_PATH_save_residual");
    }
#else
    // Compensation - save
    {
        auto &grad_residual_ptr = engine->get_grad_residual(task);
        
        // zero out selected gradients
        for (std::size_t i = 0; i < seg_idx.second; i++)
            original_src.first[seg_idx.first[i]] = 0;

        // save to residual
        /* TODO: Improve here2 */
        engine->record_stat_start(task, "CRIT_PATH_save_residual");
#if !(SKIP_SOME_CRITICAL_PATHS)
        memcpy(grad_residual_ptr.get(), original_src.first, sizeof(float) * task->tensor_numel_);
#endif
        engine->record_stat_end(task, "CRIT_PATH_save_residual");
    }
#endif
    engine->record_stat_end(task, "Compress");

    if (engine->world_size() == 1) {
        engine->schedule(engine->module_cpu_optimize_, task);
    } else {
        engine->schedule(engine->module_grad_exchange_, task);
    }

    return ModuleRunResult::RESULT_SUCCESS;
}