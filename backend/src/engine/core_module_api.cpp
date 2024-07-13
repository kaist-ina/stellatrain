
#include "core_internal.h"
#include "module.h"
#include "comm_manager.h"
#include "shm_manager.h"

void FasterDpEngine::schedule(std::unique_ptr<Module> &mod, TrainTaskV2 *task) {
    assert(task);
    assert(task->valid());

    task->state_update(TASK_PENDING);

    std::unique_lock<std::mutex> ul(lst_futures_write_mutex_);
    lst_futures_->push_back(this->thread_pool_->enqueue_priority(task->priority(), [this, &mod, task] () {
        assert(task);
        assert(task->valid());
        task->state_update(TASK_RUNNING);
        nvtxRangePush((mod->module_name() + "-" + task->key()).c_str());
        task->set_debug_message("schedule");
        mod->run(this, task);
        nvtxRangePop();
        task->state_update(TASK_IDLE);
    }));
}


void FasterDpEngine::schedule_terminate(TrainTaskV2 *task) {
    assert(task);
    assert(task->valid());

    task->state_update(TASK_FINISHED);

    std::unique_lock<std::mutex> ul(lst_futures_write_mutex_);
    lst_futures_->push_back(this->thread_pool_->enqueue_priority(task->priority(), [this, task] () {
        assert(task);
        assert(task->valid());
        const std::string key(task->key());
        free_task(task);
    }));
}


void FasterDpEngine::schedule_after_use(std::unique_ptr<Module> &mod, TrainTaskV2 *task) {
    assert(task);
    assert(task->valid());

    task->state_update(TASK_PENDING_USE);

    schedule_after_use_impl(task, [this, &mod, task] () {
        std::unique_lock<std::mutex> ul(lst_futures_write_mutex_);
        lst_futures_->push_back(this->thread_pool_->enqueue_priority(task->priority(), [this, &mod, task] () {
            assert(task);
            assert(task->valid());
            // std::cout << "Scheduing for " << task->persistent_key_ << std::endl;
            task->state_update(TASK_RUNNING);
            nvtxRangePush((mod->module_name() + "-" + task->key()).c_str());
            task->set_debug_message("schedule_after_use");
            mod->run(this, task);
            nvtxRangePop();
            task->state_update(TASK_IDLE);
        }));
    });
}


void FasterDpEngine::return_cpu_shmem_after_use(TrainTaskV2 *task) {
    std::unique_lock<std::mutex> ul(cpu_shmem_use_map_mutex_);
    auto spkey = std::make_pair(task->persistent_key(), task->iter());
    assert(cpu_shmem_use_map_.find(spkey) != cpu_shmem_use_map_.end());
    assert(cpu_shmem_use_map_[spkey] == task);
    cpu_shmem_use_map_.erase(spkey);
    cpu_shmem_use_map_cond_.notify_all();
    task->shared_cpu_tensor_entry_ = nullptr;
}

void FasterDpEngine::schedule_after_use_impl(TrainTaskV2 *task, std::function<void()> callback_fn) {
    assert(task);
    assert(task->valid());

    std::shared_ptr<std::packaged_task<void ()>> ptr_callback_fn = nullptr;
    bool schedule_immediately = false;
    auto spkey = std::make_pair(task->persistent_key(), task->iter());

    {
        std::unique_lock<std::mutex> ul(cpu_shmem_use_map_mutex_);
        if (cpu_shmem_use_map_.find(spkey) == cpu_shmem_use_map_.end()) {
            cpu_shmem_use_map_[spkey] = task;
            schedule_immediately = true;
        } else {
            ptr_callback_fn = std::make_shared<std::packaged_task<void()>>
                (std::bind(std::forward<std::function<void()>>(callback_fn)));
            cpu_shmem_use_callback_map_.insert(std::make_pair(task, ptr_callback_fn));
            // std::cout << "Queueing for " << task->key_ << std::endl;
        }
    }

    if (ptr_callback_fn) {
        std::unique_lock<std::mutex> ul(lst_futures_write_mutex_);
        lst_futures_->push_back(ptr_callback_fn->get_future());
    }

    if (schedule_immediately) {
        // std::cout << "Run immmediate " << task->persistent_key_ << std::endl;
        task->state_update(TASK_PENDING);  
        callback_fn();
    } else {
        cpu_shmem_use_map_cond_.notify_all();
    }
}



void FasterDpEngine::schedule_after_barrier_impl(TrainTaskV2 *task, std::function<void()> callback_fn, std::string barrier_name) {
    assert(task);
    assert(task->valid());
    
    bool is_finished = false;
    bool schedule_immediately = false;

    /* validity check */
    assert(task);
    assert(task->state() == TASK_PENDING_BARRIER);

    {
        std::unique_lock<std::mutex> gl(train_task_map_mutex_);
        assert(train_task_set_.find(task) != train_task_set_.end());
        std::unique_lock<std::mutex> ul(task->mutex_);
        gl.unlock();

        auto &barrier = task->shared_props_->train_barrier_[task->barrier_id_future_];
        pthread_mutex_lock(&barrier.mutex_);

        unsigned barrier_id = task->barrier_id_future_;
        assert(barrier_id < MAX_NUM_BARRIER);
        // local_state.debug_log_ << "barrier_id_future_=" << barrier_id << ", local_state=" << local_state.state_ << "\n";

        assert(barrier.state_ != FINISHED);
        /**
         * Two possible states:
         * 1. uninitialized and barrier.count == 0
         * 2. initialized, and barrier.count > 0 and barrier.count < node_world_size_ 
         * 
         * Checking barrier_name to make sure if we're using the same barriers
         * */

        if (barrier.state_ == UNINITIALIZED) {
            assert(barrier.count_ == 0);
            barrier.state_ = INITIALIZED;
            strncpy(barrier.name_, barrier_name.c_str(), MAX_BARRIER_NAME_LEN);
        } else if (barrier.state_ == INITIALIZED) {
            assert(barrier.count_ > 0);
            assert(barrier.count_ < node_world_size_);

#if CHECK_BARRIER_NAME
            // does not work well with mutiple nodes
            if (strncmp(barrier.name_, barrier_name.c_str(), MAX_BARRIER_NAME_LEN)) {
                std::cerr << "Barrier " << barrier_id << " is using name " << barrier.name_ << ", but try to apply " << barrier_name << std::endl;
                for(size_t i = 0; i <= barrier_id; i++) {
                    std::cerr << "Barrier " << i << " : " << task->shared_props_->train_barrier_[i].name_ << std::endl;
                }
            }
            assert(strncmp(barrier.name_, barrier_name.c_str(), MAX_BARRIER_NAME_LEN) == 0);
#endif
        }

        assert(barrier.state_ != UNINITIALIZED);
        barrier.count_ += 1;
        assert(barrier.count_ <= node_world_size_);

        if (barrier.count_ == node_world_size_) {
            schedule_immediately = true;
            barrier.state_ = FINISHED;
        }
        task->barrier_id_future_++;

        // local_state.debug_log_ << "barrier-state=" << barrier.state_  << ", count=" << barrier.count_ << ", schedule_immediately=" << schedule_immediately << "\n";
        assert(barrier.state_ != FINISHED || barrier.count_ == node_world_size_);
        assert(barrier.state_ != INITIALIZED || barrier.count_ < node_world_size_);
        pthread_mutex_unlock(&barrier.mutex_);

        assert(task->state() == TASK_PENDING_BARRIER);
        if (!schedule_immediately) {
            // use state machine here
            // can two local threads run concurrently? 
            task->barrier_callback_fn_[barrier_id] = std::make_shared<std::packaged_task<void()>>
                (std::bind(std::forward<std::function<void()>>(callback_fn)));
            {
                std::unique_lock<std::mutex> ul(lst_futures_write_mutex_);
                lst_futures_->push_back(task->barrier_callback_fn_[barrier_id]->get_future());
            }
        }
    }

    if (schedule_immediately) {
        task->state_update(TASK_PENDING);  
        callback_fn();
    } 
    pthread_cond_broadcast(&shared_props_->barrier_ipc_cond_);
}

void FasterDpEngine::schedule_after_cuda_impl(TrainTaskV2 *task, std::function<void()> callback_fn) {
    assert(task);
    assert(task->valid());
    const auto key(task->key());

    std::shared_ptr<std::packaged_task<void ()>> ptr_callback_fn;
    bool schedule_immediately = false;

    {
        std::unique_lock<std::mutex> ul(cuda_wait_callback_map_mutex_);
        if (cuda_wait_callback_map_.find(task) != cuda_wait_callback_map_.end()) {
            assert(cuda_wait_callback_map_[task] == nullptr);
            cuda_wait_callback_map_.erase(task);
            schedule_immediately = true;
        } else {
            ptr_callback_fn = std::make_shared<std::packaged_task<void()>>
                (std::bind(std::forward<std::function<void()>>(callback_fn)));
            cuda_wait_callback_map_.insert(std::make_pair(task, ptr_callback_fn));
        }
    }

    /**
     * Note that, at this moment, if schedule_immediately == false,
     * ptr_callback_fn might have been already fulfilled. task may not be valid afterwards. */

#if 0
    const char * train_task_state_name[] = {
        "INITIALIZED",
        "PENDING",
        "RUNNING",
        "PENDING_CUDA",
        "PENDING_BARRIER",
        "PENDING_COMM",
        "PENDING_MODEL_COMPLETE",
        "PENDING_USE",
        "FINISHED",
        "IDLE",
    };
    
    assert(task);
    assert(task->valid());
#endif
    if (ptr_callback_fn) {
        std::unique_lock<std::mutex> ul(lst_futures_write_mutex_);
        lst_futures_->push_back(ptr_callback_fn->get_future());
    }
#if 0

    std::stringstream s;
    if (!task || !task->valid()) {
        s << "=======INVALID TASK " << key << " SCHEDULED TO schedule_after_cuda_impl =======\n";
        for(auto pair = v.begin(); pair != v.end(); ++pair) {
            s << "Tried to go " << train_task_state_name[pair->first] << ", changed to " <<  train_task_state_name[pair->second] << "\n";
        }
        s << "======== EOL =========\n";
    }
    std::cerr << s.str() << std::endl;

    assert(task);
    assert(task->valid());
#endif 


    if (schedule_immediately) {
        assert(task);
        assert(task->valid());
        // std::stringstream s;
        // s << "Already CUDA finished, directly scheduling [" << task->key() << "]";
        // std::cerr << s.str() << std::endl;
        task->state_update(TASK_PENDING);  
        callback_fn();
    } else {
        // std::stringstream s;
        // s << "Not yet CUDA finished, waiting [" << task->key() << "]";
        // std::cerr << s.str() << std::endl;
    }
}

void FasterDpEngine::schedule_after_model_complete_impl(TrainTaskV2 *task, std::function<void()> callback_fn) {
    assert(task);
    assert(task->valid());

    std::shared_ptr<std::packaged_task<void ()>> ptr_callback_fn = nullptr;
    bool schedule_immediately = false;
    const std::string key(task->key());
    const std::string skey(task->persistent_key());

    {
        std::unique_lock<std::mutex> ul(layer_model_completed_version_map_mutex_);
        if (layer_model_completed_version_map_[skey] >= task->iter()) {
            schedule_immediately = true;
        } else {
            ptr_callback_fn = std::make_shared<std::packaged_task<void()>>
                (std::bind(std::forward<std::function<void()>>(callback_fn)));
            layer_model_completed_callback_map_.insert(std::make_pair(task, ptr_callback_fn));
        }
    }

    if (ptr_callback_fn) {
        std::unique_lock<std::mutex> ul(lst_futures_write_mutex_);
        lst_futures_->push_back(ptr_callback_fn->get_future());
    }

    if (schedule_immediately) {
        task->state_update(TASK_PENDING);  
        callback_fn();
    } else {
        layer_model_completed_version_map_cond_.notify_all();
    }
}


void FasterDpEngine::schedule_after_comm_impl(TrainTaskV2 *task, float * dst_val, uint32_t * dst_idx, size_t len, std::function<void()> callback_fn, unsigned iter) {
    assert(task);
    assert(task->valid());

    std::shared_ptr<std::packaged_task<void ()>> ptr_callback_fn = nullptr;
    bool schedule_immediately = false;

    auto &comm_map_mutex = comm_manager_->get_pull_callback_map_mutex();
    auto &comm_map = comm_manager_->get_pull_callback_map();
    const std::string key(task->key() + "!" + std::to_string(iter));


    RecvBufSpec recv_buf_spec;
    {
        std::unique_lock<std::mutex> ul(comm_map_mutex);
        if (comm_map.find(key) != comm_map.end()) {
            auto &f = comm_map[key];
            assert(f.first == nullptr);
            recv_buf_spec = f.second;
            comm_map.erase(key);
            schedule_immediately = true;

            // std::stringstream s;
            // s << "unstashing " << key << ", remaining " << comm_map.size();
            // std::cerr << s.str() << std::endl;
        } else {
            ptr_callback_fn = std::make_shared<std::packaged_task<void()>>
                (std::bind(std::forward<std::function<void()>>(callback_fn)));
            // std::stringstream s;
            // s << "[WAIT] " << key << " : waiting, mapsz=" << comm_map.size();
            // std::cerr << s.str() << std::endl;
            RecvBufSpec spec;
            spec.is_malloc_segment = false;
            spec.idx = dst_idx;
            spec.val = dst_val;
            spec.len = len;
            comm_map.insert(std::make_pair(key, std::make_pair(ptr_callback_fn, spec)));
        }
    }

    if (ptr_callback_fn) {
        std::unique_lock<std::mutex> ul(lst_futures_write_mutex_);
        lst_futures_->push_back(ptr_callback_fn->get_future());
    }

    if (schedule_immediately) {
        // std::cerr << "unstashing " << key << std::endl;
        // std::stringstream s;
        // s << "[WAIT] " << key << " : without wait, mapsz=" << comm_map.size();
        // std::cerr << s.str() << std::endl;
        assert(recv_buf_spec.is_malloc_segment);
        assert(recv_buf_spec.len == len);
        memcpy(dst_idx, recv_buf_spec.idx, recv_buf_spec.len * sizeof(uint32_t));
        memcpy(dst_val, recv_buf_spec.val, recv_buf_spec.len * sizeof(float));
        delete [] recv_buf_spec.idx;
        delete [] recv_buf_spec.val;
        
        task->state_update(TASK_PENDING);  
        callback_fn();
    } else {
        // std::cerr << "queueing " << key << std::endl;
    }
}

void FasterDpEngine::schedule_after_barrier(std::unique_ptr<Module> &mod, TrainTaskV2 *task, std::string barrier_name) {   
    assert(task);
    assert(task->valid());
    task->state_update(TASK_PENDING_BARRIER);

    schedule_after_barrier_impl(task, [this, &mod, task] () {
        assert(task);
        assert(task->valid());
        std::unique_lock<std::mutex> ul(lst_futures_write_mutex_);
        lst_futures_->push_back(this->thread_pool_->enqueue_priority(task->priority(), [this, &mod, task] () {
            task->state_update(TASK_RUNNING);
            nvtxRangePush((mod->module_name() + "-" + task->key()).c_str());
            task->set_debug_message("schedule_after_barrier");
            mod->run(this, task);
            nvtxRangePop();
            task->state_update(TASK_IDLE);
        }));
    }, barrier_name);
}


void FasterDpEngine::schedule_after_cuda(std::unique_ptr<Module> &mod, TrainTaskV2 *task) {    
    assert(task);
    assert(task->valid());
    std::string key_copy_for_debug(task->key());
    task->state_update(TASK_PENDING_CUDA);

    schedule_after_cuda_impl(task, [this, &mod, task] () {
        assert(task);
        assert(task->valid());
        std::unique_lock<std::mutex> ul(lst_futures_write_mutex_);
        lst_futures_->push_back(this->thread_pool_->enqueue_priority(task->priority(), [this, &mod, task] () {
            task->state_update(TASK_RUNNING);
            nvtxRangePush((mod->module_name() + "-" + task->key()).c_str());
            task->set_debug_message("schedule_after_cuda");
            mod->run(this, task);
            nvtxRangePop();
            task->state_update(TASK_IDLE);
        }));
    });
}

void FasterDpEngine::schedule_after_model_complete(std::unique_ptr<Module> &mod, TrainTaskV2 *task) {   
    assert(task);
    assert(task->valid()); 
    task->state_update(TASK_PENDING_MODEL_COMPLETE);

    schedule_after_model_complete_impl(task, [this, &mod, task] () {
        assert(task);
        assert(task->valid());
        std::unique_lock<std::mutex> ul(lst_futures_write_mutex_);
        lst_futures_->push_back(this->thread_pool_->enqueue_priority(task->priority(), [this, &mod, task] () {
            task->state_update(TASK_RUNNING);
            nvtxRangePush((mod->module_name() + "-" + task->key()).c_str());
            std::string key(task->key());
            task->set_debug_message("schedule_after_model_complete");
            mod->run(this, task);
            nvtxRangePop();
            task->state_update(TASK_IDLE);
        }));
    });
}

void FasterDpEngine::schedule_after_comm(std::unique_ptr<Module> &mod, TrainTaskV2 *task, float * dst_val, uint32_t * dst_idx, size_t len, unsigned iter) {    
    assert(task);
    assert(task->valid());
    assert(iter + 1 < world_size());
    task->state_update(TASK_PENDING_COMM);

    schedule_after_comm_impl(task, dst_val, dst_idx, len, [this, &mod, task] () {
        std::unique_lock<std::mutex> ul(lst_futures_write_mutex_);
        lst_futures_->push_back(this->thread_pool_->enqueue_priority(task->priority(), [this, &mod, task] () {
            assert(task);
            assert(task->valid());
            task->state_update(TASK_RUNNING);
            nvtxRangePush((mod->module_name() + "-" + task->key()).c_str());
            task->set_debug_message("schedule_after_comm");
            mod->run(this, task);
            nvtxRangePop();
            task->state_update(TASK_IDLE);
        }));
    }, iter);
}


void FasterDpEngine::update_model_version(TrainTaskV2 *task) {
    {
        std::unique_lock<std::mutex> ul(layer_model_map_mutex_);
        const std::string skey(task->persistent_key());
        assert(layer_model_version_map_.find(skey) != layer_model_version_map_.end());
        assert(layer_model_version_map_[skey] == std::max(0, task->iter()));
        layer_model_version_map_[skey]++;
        // std::cerr << "Updating model version of " << skey << " to " << layer_model_version_map_[skey] << std::endl;
    }
    layer_model_map_cond_.notify_all();
}


int FasterDpEngine::get_model_version(TrainTaskV2 *task) {
    std::unique_lock<std::mutex> ul(layer_model_map_mutex_);
    const std::string skey(task->persistent_key());
    return layer_model_version_map_[skey];
}

int FasterDpEngine::get_completed_model_version(TrainTaskV2 *task) {
    std::unique_lock<std::mutex> ul(layer_model_completed_version_map_mutex_);
    const std::string skey(task->persistent_key());
    return layer_model_completed_version_map_[skey];
}

TrainTaskV2 *FasterDpEngine::find_task_by_key(const std::string key) {
    std::unique_lock<std::mutex> ul(train_task_map_mutex_);
    auto it = train_task_map_.find(key);
    assert(it != train_task_map_.end());
    return it->second;
}

void FasterDpEngine::task_state_update_by_key(const std::string key, const train_task_state_t desired_state) {
    std::unique_lock<std::mutex> ul(train_task_map_mutex_);
    auto it = train_task_map_.find(key);
    assert(it != train_task_map_.end());
    
    auto task = it->second;
    assert(task);
    task->state_update(desired_state, false); /* lock is already applied */
}

void FasterDpEngine::record_stat_start(TrainTaskV2 *task, const std::string event) {
#if ENABLE_STAT    
    record_stat_impl(task, event, false);
#endif
}

void FasterDpEngine::record_stat_end(TrainTaskV2 *task, const std::string event) {
#if ENABLE_STAT    
    record_stat_impl(task, event, true);
#endif
}

std::unique_ptr<float []> &FasterDpEngine::get_grad_residual(TrainTaskV2 *task) {
    std::unique_lock<std::mutex> ul(map_compensate_grad_mutex_);
    auto skey = task->persistent_key();
    auto it = map_compensate_grad_.find(skey);
    if (it == map_compensate_grad_.end()) {
        assert(task->tensor_numel_);
        map_compensate_grad_[skey] = std::make_unique<float []>(task->tensor_numel_);
        memset(map_compensate_grad_[skey].get(), 0, sizeof(float) * task->tensor_numel_);
    }
    return map_compensate_grad_[skey];
}


torch::Tensor FasterDpEngine::tensor_from_mock(SharedCpuTensorMock &mock) {
    return shm_manager_->tensor_from(
        mock.offset, c10::IntArrayRef(mock.len, mock.len_len)
    );
}