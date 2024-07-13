#include "cpu_optimize.h"
#include "../core.h"
#include "../../optim/sparse_optimizer.h"
#include "../../optim/sgd.h"
#include "../shm_manager.h"
#include "../comm_manager.h"
#include <iomanip>

#if DEBUG_ACCURACY
static bool is_first = true;
#endif

// unique function
size_t unique1d(const torch::Tensor& input, uint32_t* pointer) {
    std::unordered_set<uint32_t> unique_elements(input.data_ptr<int32_t>(), input.data_ptr<int32_t>() + input.size(0));

    // // Assuming input is a 1D tensor of integer type
    // for (int64_t i = 0; i < input.size(0); ++i) {
    //     unique_elements.insert(static_cast<uint32_t>(input[i].item<int32_t>()));
    // }
    std::copy(unique_elements.begin(), unique_elements.end(), pointer);
    
    return unique_elements.size();
}

ModuleRunResult ModuleCpuOptimize::run(FasterDpEngine *engine, TrainTaskV2 *task) {
    LOG_DEBUG(*this, "Running " + task->key());
    assert(engine->is_node_master());
    engine->record_stat_start(task, "CpuOptimize");

    // const std::string key(task->persistent_key_);
    // std::cerr << "ModuleCpuOptimize " << task->key_ << std::endl;
    
    auto ptr_param = engine->tensor_from_mock(task->shared_cpu_tensor_entry_->param_).data_ptr<float>();
    auto param_len = task->tensor_numel_;
    auto ptr_grad_val = task->compressed_grad_val_ptr_;
    auto ptr_grad_idx = task->compressed_grad_idx_ptr_;
    auto ptr_grad_len = task->tensor_compressed_numel_;

#if MERGE
    torch::Tensor merged_grad = torch::zeros({int(param_len)});
    unsigned int merge_offset = 0;
    unsigned int step_size = ptr_grad_len / engine->world_size();
    std::vector<torch::Tensor> idx_tensor;

    for (unsigned int i = 0; i < engine->world_size(); i++) {
        torch::Tensor tmp_grad = torch::zeros({int(param_len)});
        torch::Tensor tmp_idx = torch::from_blob(ptr_grad_idx+merge_offset, static_cast<int64_t>(step_size), torch::TensorOptions().dtype(torch::kInt32));
        torch::Tensor tmp_values = torch::from_blob(ptr_grad_val+merge_offset, static_cast<int64_t>(step_size), torch::TensorOptions().dtype(torch::kFloat32));
        tmp_grad.index_put_({tmp_idx.to(torch::kLong)}, tmp_values);
        idx_tensor.push_back(tmp_idx.clone());
        merged_grad += tmp_grad;
        merge_offset += step_size;
    }
    merged_grad /= float(engine->world_size());
    
    torch::Tensor concated_idx = torch::cat(idx_tensor);
    auto unique_len = unique1d(concated_idx, ptr_grad_idx);
    torch::Tensor unique_idx = torch::from_blob(ptr_grad_idx, static_cast<int64_t>(unique_len), torch::TensorOptions().dtype(torch::kInt32)).to(torch::kLong);

    torch::Tensor compressed_merged_grad = merged_grad.index_select(0, unique_idx);

    std::copy(compressed_merged_grad.data_ptr<float>(), 
          compressed_merged_grad.data_ptr<float>() + compressed_merged_grad.numel(), 
          ptr_grad_val);
          
    ptr_grad_len = unique_idx.numel();
    // optional 
    //
    //
    assert(unique_idx.numel() == compressed_merged_grad.numel());
#endif 

    assert(ptr_grad_val);
    assert(ptr_grad_idx);

#if DEBUG_ACCURACY
    assert(engine->sparse_optimizer());
    if (is_first) {
        is_first = false;
        std::cerr << "Using Sparse Optimizer: " << engine->sparse_optimizer()->name() << std::endl;
        std::cerr << "Using LR / Momentum : " << dynamic_cast<SGD &>(*engine->sparse_optimizer()).get_lr() << " / " << dynamic_cast<SGD &>(*engine->sparse_optimizer()).get_momentum() << std::endl;
   }

    for(unsigned i = 0; i < ptr_grad_len; i++) {
        assert(ptr_grad_idx[i] < param_len);
        assert(!isnan(ptr_grad_val[i]));
    }
#endif

#if DEBUG_ACCURACY
    auto tensor_param = torch::from_blob(ptr_param, param_len);
    auto tensor_grad = torch::from_blob(ptr_grad_val, ptr_grad_len);
    // auto tensor_pos = torch::from_blob(ptr_grad_idx, ptr_grad_len, at::TensorOptions().dtype(torch::kInt32));
    std::stringstream s;
    if (task->persistent_key() == "14@bias") {
        s << "[" << task->iter() << "] " << std::setprecision (15) << task->persistent_key() << ": param (" << tensor_param.norm().item<float>() << ") [" << param_len << "] + grad (" << tensor_grad.norm().item<float>() << ") ["<< tensor_grad.numel() << "] = ";
    }
    float norm_before = tensor_grad.norm().item<float>();

#endif

    engine->record_stat_start(task, "CRIT_PATH_optimize_raw");
#if !(SKIP_SOME_CRITICAL_PATHS)
    engine->sparse_optimizer()->optimize_raw(ptr_param, param_len, task->persistent_key(), 
         ptr_grad_val, ptr_grad_idx, ptr_grad_len);
#endif
    engine->record_stat_end(task, "CRIT_PATH_optimize_raw");

#if DEBUG_ACCURACY    
    float norm_after = tensor_grad.norm().item<float>();
    if (norm_before != norm_after) {
        std::stringstream s;
        s << "[" << task->iter() << "] " << task->persistent_key() << ": Data corrupt during optimization. Expected norm = " << norm_before << ", but got " << norm_after << ".";
        std::cerr << s.str() << std::endl;
        assert(norm_before == norm_after);
    }

#endif

#if DEBUG_ACCURACY
    for(unsigned i = 0; i < param_len; i++) {
        assert(!isnan(ptr_param[i]));
    }
#endif


#if PRIORITY_TX
    engine->comm_manager()->delegate_delete(task);
#else
    delete [] task->compressed_grad_val_ptr_;
    delete [] task->compressed_grad_idx_ptr_;
#endif

    engine->record_stat_end(task, "CpuOptimize");
    engine->record_stat_start(task, "CpuOptimizeBarrier");
    engine->schedule_after_barrier(engine->module_h2d_copy_pre_, task, module_name());

    return ModuleRunResult::RESULT_SUCCESS;
}