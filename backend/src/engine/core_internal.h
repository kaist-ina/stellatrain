#ifndef _ENGINE_CORE_INTERNAL_H_
#define _ENGINE_CORE_INTERNAL_H_

#include <cstdint>
#include <pthread.h>
#include <assert.h>
#include "core.h"
#include "task.h"


#if PROFILER_ENABLED
#include <nvToolsExt.h>
#else
#define nvtxRangePush(x)
#define nvtxRangePop(x)
#endif 

#define assert_p(expr) (static_cast <bool> (expr) ? void (0) : __assert_perror_fail (errno, __FILE__, __LINE__, __ASSERT_FUNCTION))


static const size_t NUM_MAX_CONCURRENT_TASKS = 1024*32;
static const size_t NUM_MAX_LAYERS = 1024;

struct CpuParamEntry {
    char name[MAX_TASK_NAME_LEN+1];
    off64_t offset;
    int64_t len[NUM_MAX_PARAM_DIM];
    uint32_t len_len;
};

struct SharedCoreProps {

    // ring buffer for tasks
    off_t task_buffer_head_;
    off_t task_buffer_tail_;
    uint32_t task_alloc_num_;
    TrainTaskSharedProps task_buffer_[NUM_MAX_CONCURRENT_TASKS];

    uint64_t canary_;

    CpuParamEntry cpu_param_lst_[NUM_MAX_LAYERS];
    CpuParamEntry cpu_grad_lst_[NUM_MAX_LAYERS];
    uint32_t cpu_param_num_;
    uint32_t cpu_grad_num_;
    
    SharedCpuTensorEntry cpu_tensor_lst_[NUM_MAX_LAYERS];
    uint32_t cpu_tensor_num_;


    pthread_mutex_t global_ipc_mutex_;
    
    pthread_mutex_t barrier_ipc_mutex_;
    pthread_cond_t barrier_ipc_cond_;

};



#endif