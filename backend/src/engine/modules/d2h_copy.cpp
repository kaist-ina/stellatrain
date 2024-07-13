#include "d2h_copy.h"
#include "../core.h"

#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAEvent.h>
#include <c10/cuda/CUDAStream.h>

#if PROFILER_ENABLED
#include <nvToolsExt.h>
#else
#define nvtxRangePush(x)
#define nvtxRangePop(x)
#endif 

#if DEBUG_ACCURACY
static bool is_first = true;
#endif


ModuleRunResult ModuleD2HCopy::run(FasterDpEngine *engine, TrainTaskV2 *task) {
    LOG_DEBUG(*this, "Running " + task->key());
    assert(task);

    engine->record_stat_start(task, "Total");
    engine->record_stat_start(task, "D2HCopy");

    // auto &cpu_grad_tensor = task->shared_cpu_tensor_entry_->grad_-
    auto &gpu_grad_tensor = task->gpu_grad_tensor();
    auto cpu_grad_tensor = engine->tensor_from_mock(task->shared_cpu_tensor_entry_->grad_[engine->local_rank()]);

    CudaCallbackUserData *cuda_cb_data = reinterpret_cast<CudaCallbackUserData *>(malloc(sizeof(CudaCallbackUserData)));
    cuda_cb_data->engine = engine;
    cuda_cb_data->task = task;


    #if DEBUG_ACCURACY
    
    // if(torch::isnan(gpu_grad_tensor).any().item<bool>()) {
    //     LOG_ERROR(*this, "NaN detected in GPU tensor");
    //     assert(false);
    // }
    #endif
    
    cudaEvent_t cuda_event;
    C10_CUDA_CHECK(cudaEventCreate(&cuda_event));
    C10_CUDA_CHECK(cudaEventRecord(cuda_event));

    {
        torch::NoGradGuard no_grad;
        const auto & stream = vec_stream_gpu_to_cpu_copy_[worker_id];
        at::cuda::CUDAStreamGuard stream_guard(stream);

        C10_CUDA_CHECK(cudaStreamWaitEvent(stream, cuda_event));
        cpu_grad_tensor.copy_(gpu_grad_tensor, true);
        gpu_grad_tensor.record_stream(stream);
        
        // report certain task is completed
        // https://stackoverflow.com/questions/64979087/wait-until-any-device-has-finished-in-cuda
        C10_CUDA_CHECK(cudaLaunchHostFunc(stream, FasterDpEngine::report_cuda_finished, cuda_cb_data));
    }

    engine->schedule_after_cuda(engine->module_d2h_copy_post_, task);


    return ModuleRunResult::RESULT_SUCCESS;
}


ModuleRunResult ModuleD2HCopyPost::run(FasterDpEngine *engine, TrainTaskV2 *task) {
    LOG_DEBUG(*this, "Running " + task->key());

    engine->record_stat_end(task, "D2HCopy");
    engine->record_stat_start(task, "D2HCopyBarrier");

    auto &gpu_grad_tensor = task->gpu_grad_tensor();
    {
        torch::NoGradGuard no_grad;
        gpu_grad_tensor.zero_();
        task->free_gpu_grad_tensor();
    }

    #if DEBUG_ACCURACY
        auto cpu_grad_tensor = engine->tensor_from_mock(task->shared_cpu_tensor_entry_->grad_[engine->local_rank()]);

        float *ptr = reinterpret_cast<float *>(cpu_grad_tensor.data_ptr());
        for (unsigned i = 0; i < task->tensor_numel_; i++) {
            assert(!isnan(ptr[i]));
        }
    #endif

    engine->schedule_after_barrier(engine->module_cpu_gather_, task, module_name());

    return ModuleRunResult::RESULT_SUCCESS;
}