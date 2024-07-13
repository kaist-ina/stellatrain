#include "h2d_copy.h"
#include "../core.h"

#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAEvent.h>

#if PROFILER_ENABLED
#include <nvToolsExt.h>
#else
#define nvtxRangePush(x)
#define nvtxRangePop(x)
#endif 


ModuleRunResult ModuleH2DCopyPre::run(FasterDpEngine *engine, TrainTaskV2 *task) {
    LOG_DEBUG(*this, "Running " + task->key());

    engine->record_stat_end(task, "CpuOptimizeBarrier");
    engine->record_stat_start(task, "H2DCopyPre");
    
    // we need to wait until the use of previous model is finished
#if SKIP_SOME_CRITICAL_PATHS
    engine->schedule_after_model_complete(engine->module_h2d_copy_post_, task);
#else
    engine->schedule_after_model_complete(engine->module_h2d_copy_, task);
#endif
    return ModuleRunResult::RESULT_SUCCESS;
}


ModuleRunResult ModuleH2DCopy::run(FasterDpEngine *engine, TrainTaskV2 *task) {
    LOG_DEBUG(*this, "Running " + task->key());
    engine->record_stat_end(task, "H2DCopyPre");
    engine->record_stat_start(task, "H2DCopy");



    auto cpu_param_tensor = engine->tensor_from_mock(task->shared_cpu_tensor_entry_->param_);
    auto &gpu_param_tensor = task->gpu_param_tensor();

    CudaCallbackUserData *cuda_cb_data = reinterpret_cast<CudaCallbackUserData *>(malloc(sizeof(CudaCallbackUserData)));
    cuda_cb_data->engine = engine;
    cuda_cb_data->task = task;

    {
        torch::NoGradGuard no_grad;
        const auto & stream = vec_stream_cpu_to_gpu_copy_[worker_id];
        at::cuda::CUDAStreamGuard stream_guard(stream);
        // std::cerr << "CPU param tensor " << task->key() << "from shared entry " << task->shared_cpu_tensor_entry_ << ", size=" << cpu_param_tensor.sizes() << " " << cpu_param_tensor.data_ptr<float>() << std::endl;
        gpu_param_tensor.copy_(cpu_param_tensor, true);

        C10_CUDA_CHECK(cudaLaunchHostFunc(stream, FasterDpEngine::report_cuda_finished, cuda_cb_data));
    }
    engine->schedule_after_cuda(engine->module_h2d_copy_post_, task);
    return ModuleRunResult::RESULT_SUCCESS;
}


ModuleRunResult ModuleH2DCopyPost::run(FasterDpEngine *engine, TrainTaskV2 *task) {
    LOG_DEBUG(*this, "Running " + task->key());

    engine->record_stat_end(task, "H2DCopy");
    task->free_gpu_param_tensor();
    engine->update_model_version(task);
    engine->record_stat_end(task, "Total");
    const auto key(task->key());
    LOG_DEBUG(*this, "Finished " + key);
    engine->return_cpu_shmem_after_use(task);
    engine->schedule_terminate(task);

    return ModuleRunResult::RESULT_SUCCESS;
}