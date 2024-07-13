#include "task.h"
#include "logger.h"
#include "core.h"
#include <cstring>
#include <cassert>
#include <iostream>
#include <torch/torch.h>


void SharedCpuTensorEntry::initialize(const std::string &skey, int dup_iter_idx) {

    assert(!valid_);
    valid_ = true;

    memset(persistent_key_, 0, MAX_TASK_NAME_LEN+1);
    strncpy(persistent_key_, skey.c_str(),MAX_TASK_NAME_LEN);
    memset(&param_, 0, sizeof(SharedCpuTensorMock));
    memset(grad_, 0, MAX_NUM_GPU_PER_NODE * sizeof(SharedCpuTensorMock));
    entry_dup_idx_ = dup_iter_idx;

    param_version_ = -1;
    for (size_t i = 0; i < MAX_NUM_GPU_PER_NODE; i++) {
        grad_version_[i] = -1;
    }
}

FasterDpEngine *TrainTaskV2::engine_ = nullptr;


void TrainTaskV2::set_engine(FasterDpEngine *engine) {
    engine_ = engine;
}

TrainTaskV2::TrainTaskV2(unsigned iter_cnt, int layer, const std::string key_str) 
    : valid_(true), state_(TASK_INITIALIZED), barrier_id_(0), barrier_id_future_(0), 
    tensor_numel_(0), 
    tensor_compressed_numel_(0), 
    grad_sync_iter_(0),
    compressed_grad_val_ptr_(nullptr),
    compressed_grad_idx_ptr_(nullptr),
    iter_(iter_cnt), key_(to_key(iter_cnt, layer, key_str)), persistent_key_(to_persistent_key(layer, key_str)), 
    shared_props_(nullptr), shared_cpu_tensor_entry_(nullptr), test_field_(0), priority_(iter_cnt*1000+layer) { 

    if (key_.length() > MAX_TASK_NAME_LEN || persistent_key_.length() > MAX_TASK_NAME_LEN) {
        throw std::runtime_error(std::string("The length of layer name must not exceed ") + std::to_string(MAX_TASK_NAME_LEN) + ".");
    }
}

std::string TrainTaskV2::to_key(unsigned iter_cnt, int layer, const std::string &key_str) {
    char key[MAX_TASK_NAME_LEN+1];
    snprintf(key, MAX_TASK_NAME_LEN, "%d@%d@%s", iter_cnt, layer, key_str.c_str());

    return std::string(key);
}

std::string TrainTaskV2::to_persistent_key(int layer, const std::string &key_str) {
    char key[MAX_TASK_NAME_LEN+1];
    snprintf(key, MAX_TASK_NAME_LEN, "%d@%s", layer, key_str.c_str());

    return std::string(key);
}


void TrainTaskV2::state_update(const train_task_state_t desired_state, bool apply_lock) {
    if (apply_lock) {
        std::unique_lock<std::mutex> ul(engine_->train_task_map_mutex_);
        state_update_impl(desired_state);
    } else {
        state_update_impl(desired_state);
    }
}

void TrainTaskV2::state_update_impl(const train_task_state_t desired_state) {

    // {
    if (engine_->train_task_set_.find(this) == engine_->train_task_set_.end()) {
        assert(desired_state == TASK_IDLE);
        return;
    }
    // here, task may be invalid but task cannot be deleted
    // }

    if (desired_state == TASK_IDLE && !valid()) {
        // this task has been finished, already removed from local state map
        return;
    }
    std::unique_lock<std::mutex> ul2(mutex_);

    // std::unique_lock<std::mutex> ul(mutex_);
    // assert(valid());

    const auto current_state = state_;

    if (desired_state == TASK_IDLE) {
        if (current_state == TASK_RUNNING)
            state_ = TASK_IDLE;
    } else if (desired_state == TASK_PENDING) {
        assert(current_state != TASK_PENDING);
        state_ = TASK_PENDING;
    } else if (desired_state == TASK_RUNNING) {
        assert(current_state == TASK_PENDING);
        state_ = TASK_RUNNING;
    } else if (desired_state == TASK_PENDING_CUDA) {
        assert(current_state == TASK_RUNNING || current_state == TASK_IDLE);
        state_ = TASK_PENDING_CUDA;
    }  else if (desired_state == TASK_PENDING_COMM) {
        assert(current_state == TASK_RUNNING || current_state == TASK_IDLE);
        state_ = TASK_PENDING_COMM;
    }  else if (desired_state == TASK_PENDING_MODEL_COMPLETE) {
        assert(current_state == TASK_RUNNING || current_state == TASK_IDLE);
        state_ = TASK_PENDING_MODEL_COMPLETE;
    } else if (desired_state == TASK_PENDING_BARRIER) {
        assert(current_state == TASK_RUNNING || current_state == TASK_IDLE);
        state_ = TASK_PENDING_BARRIER;
    } else if (desired_state == TASK_PENDING_USE) {
        assert(current_state == TASK_RUNNING || current_state == TASK_IDLE || current_state == TASK_INITIALIZED);
        state_ = TASK_PENDING_USE;
    } else if (desired_state == TASK_FINISHED) {
        assert(current_state == TASK_RUNNING || current_state == TASK_IDLE);
        state_ = TASK_FINISHED;
    } else {
        assert(false); // should not reach here
    }
    state_history_.push_back(std::make_pair(desired_state, state_));
}

void TrainTaskV2::assign_gpu_tensor(const torch::Tensor &param_tensor, const torch::Tensor &grad_tensor) {
    gpu_param_tensor_ = param_tensor;
    gpu_grad_tensor_ = grad_tensor;
}


void TrainTaskV2::free_gpu_grad_tensor() {
    gpu_grad_tensor_.reset();
}
void TrainTaskV2::free_gpu_param_tensor() {
    gpu_param_tensor_.reset();
}

void TrainTaskV2::prepare_delete() {
    assert(!valid_);
    
    gpu_param_tensor_.reset();
    gpu_grad_tensor_.reset();
    state_history_.clear();

    debug_log_.clear();
    debug_msg_.clear();

    shared_props_ = nullptr;
    shared_cpu_tensor_entry_ = nullptr;

}

void TrainTaskSharedProps::initialize(const std::string &key) {
    /* This function is not thread-safe */
    LOG_DEBUG(*this, "Initializing shared props for " + key);

    assert(!valid_);
    valid_ = true;

    memset(key_, 0, MAX_TASK_NAME_LEN+1);
    strncpy(key_, key.c_str(), MAX_TASK_NAME_LEN);

    /* Make mutex shared */
    {
        int ret;
        pthread_mutexattr_t mtx_attr;
        pthread_mutexattr_init(&mtx_attr);
        ret = pthread_mutexattr_setpshared(&mtx_attr, PTHREAD_PROCESS_SHARED);
        assert(ret == 0);
        ret = pthread_mutex_init(&mutex_, &mtx_attr);
        assert(ret == 0);
    }

    for (size_t i = 0; i < MAX_NUM_GPU_PER_NODE; i++) {
        is_finished_[i] = false;
        shared_cpu_data_ready_[i] = false;
        shared_test_field_[i] = 0;
    }
    finish_initiated_ = false;

    /* Initialize barriers */
    pthread_mutex_lock(&mutex_);
    int ret;
    pthread_mutexattr_t mtx_attr;
    pthread_mutexattr_init(&mtx_attr);
    ret = pthread_mutexattr_setpshared(&mtx_attr, PTHREAD_PROCESS_SHARED);
    assert(ret == 0);
        
    for (size_t i = 0; i < MAX_NUM_BARRIER; i++) {
        train_barrier_[i].count_ = 0;
        train_barrier_[i].state_ = UNINITIALIZED;
        ret = pthread_mutex_init(&train_barrier_[i].mutex_, &mtx_attr);
        assert(ret == 0);
    }
    pthread_mutex_unlock(&mutex_);
}

void TrainTaskSharedProps::lock() {
    pthread_mutex_lock(&mutex_);
}

void TrainTaskSharedProps::unlock() {
    pthread_mutex_unlock(&mutex_);
}
