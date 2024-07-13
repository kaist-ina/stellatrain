#include "core.h"
#include "core_internal.h"
#include "../compress/thresholdv.h"
#include "../compress/thresholdv16.h"
#include "../compress/topk.h"
#include "../optim/sparse_optimizer.h"
#include <iostream>
#include <csignal>
#include <pthread.h>
#include <nlohmann/json.hpp>
#include <future>
#include <chrono>
#include "task.h"
#include "threadpool.h"
#include "shm_manager.h"
#include "comm_manager.h"
#include "module.h"
#include "../misc/ipaddr.h"
// optimizer
#include "../optim/sgd.h"
#include "../optim/adam.h"

FasterDpEngine::FasterDpEngine() : finished_(false), model_staleness_(1), compression_ratio_(0.99), first_backward_(true), gradient_accumulation_(1) {
    std::cout << "Starting FasterDPEngine" << std::endl;
    configure_compression("thresholdv16");
}

FasterDpEngine::~FasterDpEngine() {
    std::cout << "Terminating FasterDPEngine" << std::endl;
    
    finished_ = true;

    if (barrier_manager_thread_ != nullptr) {
        pthread_cond_broadcast(&shared_props_->barrier_ipc_cond_);
        barrier_manager_thread_->join();
    }

    if (chore_manager_thread_ != nullptr) {
        finished_cond_.notify_all();
        chore_manager_thread_->join();
    }

    if (model_complete_manager_thread_ != nullptr) {
        layer_model_completed_version_map_cond_.notify_all();
        model_complete_manager_thread_->join();
    }

    if (cpu_shmem_return_manager_thread_ != nullptr) {
        cpu_shmem_use_map_cond_.notify_all();
        cpu_shmem_return_manager_thread_->join();
    }

    if (backward_delegate_thread_ != nullptr) {
        backward_delegate_cond_.notify_all();
        backward_delegate_thread_->join();
    }

#if ENABLE_STAT
    stat_export();
#endif
}


void FasterDpEngine::configure(std::string master_addr, uint16_t master_port, int world_size, int rank, int local_session_id, int local_world_size, int local_rank, const std::string method, int gradient_accumulation) {
    
    local_session_id_ = local_session_id;
    rank_ = rank;
    world_size_ = world_size;
    master_addr_ = master_addr;
    master_port_ = master_port;

    node_world_size_ = local_world_size;
    node_master_ = local_rank == 0;
    local_rank_ = local_rank;

    cudaSetDevice(local_rank_);
    
    master_ = rank == 0 && node_master_;

    model_staleness_ = 1;
    if (model_staleness_ != 1) {
        std::cerr << "WARNING!!!!!!!!!!!!!! MODEL STALENESS IS NOT 1!!!!!!!" << std::endl;
    }
    gradient_accumulation_ = gradient_accumulation;

    if (node_master_)
        std::cerr << "Master Endpoint: " << master_addr_ << ":" << master_port_ << " (master=" << (master_ ? "true" : "false") << ")" << std::endl;

    TrainTaskV2::set_engine(this);

    thread_pool_ = std::make_unique<ThreadPool>(node_master_ ? NUM_WORKER_THREADS_MASTER : NUM_WORKER_THREADS);

    shm_manager_ = std::make_unique<ShmManager>(node_master_, local_session_id, local_rank_);

    comm_manager_ = std::make_unique<CommManager>(master_addr_, master_port_, this);

    lst_futures_ = std::make_unique<std::list<std::future<void>>>();

    const auto meta_header = shm_manager_->meta_header();

    if (node_master_) {
        shared_props_ = reinterpret_cast<SharedCoreProps *>(shm_manager_->malloc_meta(sizeof(SharedCoreProps)));
        memset(shared_props_, 0, sizeof(SharedCoreProps));

        /* shared_props_ must be the first allocated application level metadata */
        assert(reinterpret_cast<SharedCoreProps *>(meta_header + 1) == shared_props_);
    }
    

    if (method == "thresholdv") {
        compressor_ = std::make_unique<ThresholdvCompressor>(this->thread_pool_, true);
    } else if (method == "thresholdv16") {
        compressor_ = std::make_unique<ThresholdvCompressor16>(this->thread_pool_, true);
    } else if (method == "topk") {
        compressor_ = std::make_unique<TopkCompressor>(this->thread_pool_);
    }else {
        throw std::runtime_error(std::string("Unknown compression method ") + method + ".");
    }

    load_modules();

    if (node_master_) {
        /* initialize ipc mutex and cv for barrier */
        int ret;
        pthread_mutexattr_t mutex_attr;
        memset(&mutex_attr, 0, sizeof(pthread_mutexattr_t));
        ret = pthread_mutexattr_setpshared(&mutex_attr, PTHREAD_PROCESS_SHARED);
        assert_p(ret == 0);
        ret = pthread_mutex_init(&shared_props_->barrier_ipc_mutex_, &mutex_attr);
        assert_p(ret == 0);
        ret = pthread_mutex_init(&shared_props_->global_ipc_mutex_, &mutex_attr);
        assert_p(ret == 0);
        
        pthread_condattr_t cond_attr;
        memset(&cond_attr, 0, sizeof(pthread_condattr_t));
        ret = pthread_condattr_setpshared(&cond_attr, PTHREAD_PROCESS_SHARED);
        assert_p(ret == 0);
        ret = pthread_cond_init(&shared_props_->barrier_ipc_cond_, &cond_attr);
        assert_p(ret == 0);

        /* initialize task allocator */
        shared_props_->task_buffer_head_ = 0;
        shared_props_->task_buffer_tail_ = 0;
        shared_props_->task_alloc_num_ = 0;
    }
    

    /* here wait for inter-node communicator to be established */
    if (world_size_ > 1 && node_master_) {
        if (master_)
            comm_manager_->startServer(world_size_);
        comm_manager_->startClient(rank_);

        if (master_) {
            std::cerr << "Waiting until client connect" << std::endl;
            comm_manager_->waitClientConnect();
        }
        std::cerr << "StartServ/Client OK" << std::endl;
    }

    /* barrier here to synchonize master and slave */

    shm_manager_->barrier(node_world_size_);

    /* register ptr of shared_props_ to slave */
    if (!node_master_) {
        shared_props_ = reinterpret_cast<SharedCoreProps *>(meta_header + 1);
    }

    barrier_manager_thread_ = std::make_unique<std::thread>(&FasterDpEngine::barrier_manager_thread_main, this);
    chore_manager_thread_ = std::make_unique<std::thread>(&FasterDpEngine::chore_manager_thread_main, this);
    model_complete_manager_thread_ = std::make_unique<std::thread>(&FasterDpEngine::model_complete_manager_thread_main, this);
    cpu_shmem_return_manager_thread_ = std::make_unique<std::thread>(&FasterDpEngine::cpu_shmem_return_manager_thread_main, this);
    backward_delegate_thread_ = std::make_unique<std::thread>(&FasterDpEngine::backward_delegate_thread_main, this);
    ready_ = true;

#if IS_SPEED_AFFECTED
    std::cerr << "Warning: One of debugging flags are enabled. May affect speed." << std::endl;
#endif
#if IS_ACCURACY_AFFECTED
    std::cerr << "Warning: One of debugging flags are enabled. May affect accuracy." << std::endl;
#endif
}

void FasterDpEngine::configure_compression(const std::string &method) {
    
    if (method == "thresholdv") {
        compressor_ = std::make_unique<ThresholdvCompressor>(this->thread_pool_, false);
    } else if (method == "thresholdv16") {
        compressor_ = std::make_unique<ThresholdvCompressor16>(this->thread_pool_, false);
    } 
    else {
        throw std::runtime_error(std::string("Unknown compression method ") + method + ".");
    }
}

void FasterDpEngine::configure_compression_ratio(double ratio) {
    assert(ratio > 0 && ratio <= 1);
    compression_ratio_ = ratio;
}

void FasterDpEngine::barrier() {
    shm_manager_->barrier(node_world_size_);
}

void FasterDpEngine::get_sparse_optimizer(const std::string &optimizer) {

    if (optimizer == "sgd") {
        sparse_optimizer_ = std::make_unique<SGD>();
    } else if (optimizer == "adam") {
        sparse_optimizer_ = std::make_unique<Adam>();
    } else {
        throw std::runtime_error(std::string("Unknown optimizer ") + optimizer + ".");
    }

}

// https://kyungpyo-kim.github.io/study/thread-safety-of-unordered_map/
using mutex_type = std::shared_timed_mutex;
using read_only_lock  = std::shared_lock<mutex_type>;
using updatable_lock = std::unique_lock<mutex_type>;


void CUDART_CB FasterDpEngine::report_cuda_finished(void *userData) {
    CudaCallbackUserData *cuda_cb_data = reinterpret_cast<CudaCallbackUserData *>(userData);
    auto engine = cuda_cb_data->engine;
    auto task = cuda_cb_data->task;
    free(cuda_cb_data);

    // std::stringstream s;
    // s << "Finished CUDA for " << task->key_;
    // std::cerr << s.str() << std::endl;

    engine->report_cuda_finished_impl(task);
}

/**
 * https://stackoverflow.com/questions/64979087/wait-until-any-device-has-finished-in-cuda
*/
void FasterDpEngine::report_cuda_finished_impl(TrainTaskV2 *task) {
    std::stringstream s;
    std::shared_ptr<std::packaged_task<void ()>> ptr_callback_fn;

    {
        std::unique_lock<std::mutex> ul(cuda_wait_callback_map_mutex_);
        if (cuda_wait_callback_map_.find(task) != cuda_wait_callback_map_.end()) {
            ptr_callback_fn = cuda_wait_callback_map_[task];
            cuda_wait_callback_map_.erase(task);
            
            // std::stringstream s;
            // s << "Report Cuda Finished Late, [" << task->key() << "]";
            // std::cerr << s.str() << std::endl;
        } else {
            cuda_wait_callback_map_[task] = nullptr;
            // std::stringstream s;
            // s << "Report Cuda Finished First, [" << task->key() << "]";
            // std::cerr << s.str() << std::endl;
        }
    }

    if (ptr_callback_fn) {
        task->state_update(TASK_PENDING);
        (*ptr_callback_fn)();
    }
}

int FasterDpEngine::cpu_shmem_return_manager_thread_main() {
    sigset_t mask;
    sigemptyset (&mask);
    sigaddset (&mask, SIGTERM);
    sigaddset (&mask, SIGHUP);
    pthread_sigmask(SIG_BLOCK, &mask, NULL);
    pthread_setname_np(pthread_self(), "CpuShmRManger");

    while(!finished_) {
        std::unique_lock<std::mutex> ul(cpu_shmem_use_map_mutex_);
        
        std::shared_ptr<std::packaged_task<void ()>> ptr_callback_fn = nullptr;
        TrainTaskV2 *task = nullptr;

        for (auto it = cpu_shmem_use_callback_map_.begin(); it != cpu_shmem_use_callback_map_.end(); ++it) {
            task = it->first;
            auto spkey = std::make_pair(task->persistent_key(), task->iter());
            if (cpu_shmem_use_map_.find(spkey) == cpu_shmem_use_map_.end()) {
                ptr_callback_fn = std::move(it->second);
                cpu_shmem_use_map_[spkey] = task;
                cpu_shmem_use_callback_map_.erase(it);
                break;
            }
        }

        if (ptr_callback_fn) {
            task->state_update(TASK_PENDING);
            (*ptr_callback_fn)();
        } else {
            // layer_model_completed_version_map_cond_.wait(ul);
            cpu_shmem_use_map_cond_.wait_for(ul, std::chrono::seconds(20));
        }
    }

    return 0;
}
int FasterDpEngine::model_complete_manager_thread_main() {
    sigset_t mask;
    sigemptyset (&mask);
    sigaddset (&mask, SIGTERM);
    sigaddset (&mask, SIGHUP);
    pthread_sigmask(SIG_BLOCK, &mask, NULL);
    pthread_setname_np(pthread_self(), "MdlCplManager");

    while(!finished_) {
        std::unique_lock<std::mutex> ul(layer_model_completed_version_map_mutex_);
        
        std::shared_ptr<std::packaged_task<void ()>> ptr_callback_fn = nullptr;
        TrainTaskV2 *task = nullptr;

        for (auto it = layer_model_completed_callback_map_.begin(); it != layer_model_completed_callback_map_.end(); ++it) {
            task = it->first;
            const std::string skey(task->persistent_key());
            assert(layer_model_completed_version_map_.find(skey) != layer_model_completed_version_map_.end());
            const auto cvm = layer_model_completed_version_map_[skey];
            if (cvm >= task->iter()) {
                ptr_callback_fn = std::move(it->second);
                layer_model_completed_callback_map_.erase(it);
                break;
            } else {
                // std::cerr << "model_complete_manager " << skey << "cvm=" << layer_model_completed_version_map_[skey] << ", target=" << task->iter() << std::endl;
            }
        }
        // std::cerr << "============================" << std::endl;
        if (ptr_callback_fn) {
            task->state_update(TASK_PENDING);
            (*ptr_callback_fn)();
        } else {
            // layer_model_completed_version_map_cond_.wait(ul);
            layer_model_completed_version_map_cond_.wait_for(ul, std::chrono::seconds(20));
        }
    }

    return 0;
}


int FasterDpEngine::chore_manager_thread_main() {
    sigset_t mask;
    sigemptyset (&mask);
    sigaddset (&mask, SIGTERM);
    sigaddset (&mask, SIGHUP);
    pthread_sigmask(SIG_BLOCK, &mask, NULL);
    pthread_setname_np(pthread_self(), "ChoreManager");

    while(!finished_) {
        std::unique_ptr<std::list<std::future<void>>> lst_futures_to_process;
        {
            std::unique_lock<std::mutex> ul(lst_futures_write_mutex_); // stop writing to lst_futures_!

            lst_futures_to_process = std::move(lst_futures_);
            lst_futures_ = std::make_unique<std::list<std::future<void>>>();
            ul.unlock();

            size_t cnt = 0;
            for(auto &fut : *lst_futures_to_process) {
                if (fut.wait_for(std::chrono::seconds(0)) != std::future_status::ready) {
                    std::unique_lock<std::mutex> ul2(lst_futures_write_mutex_);
                    lst_futures_->push_back(std::move(fut));
                } else {
                    fut.get(); /** Make sure exceptions are thrown */
                }
                cnt++;
            }

            lst_futures_to_process->clear();
        }

        {
            std::mutex dummy;
            std::unique_lock<std::mutex> ul(dummy);
            finished_cond_.wait_for(ul, std::chrono::seconds(10));
#if DEBUG_BARRIER
            stat_export();
#endif
        }
    }

    return 0;
}
/**
 * Worker Barrier Thread. Manages sleeping threads, and enqueue to worker queue if barrier satisfies condition
*/
int FasterDpEngine::barrier_manager_thread_main() {
    sigset_t mask;
    sigemptyset (&mask);
    sigaddset (&mask, SIGTERM);
    sigaddset (&mask, SIGHUP);
    pthread_sigmask(SIG_BLOCK, &mask, NULL);
    pthread_setname_np(pthread_self(), "BarrierManager");

    while (!finished_) {
        
        std::vector<std::pair<TrainTaskV2 *, unsigned>> tasks_to_wake;

        /* iterate with lock to invalidate states */
        {
            std::unique_lock<std::mutex> ul(train_task_map_mutex_);
            for (auto it = train_task_map_.begin(); it != train_task_map_.end();) {
                if(!it->second->valid()) {
#if DEBUG_STATE_CHANGE
                    std::stringstream s;
                    s << "Freeing local state for task=" << it->second->key();
                    std::cerr << s.str() << std::endl;
#endif
                    it->second->mutex_.lock();
                    train_task_set_.erase(it->second);
                    it->second->prepare_delete();
                    it->second->mutex_.unlock();
#if !(HACK_SKIP_FREE_TRAIN_TASK)
                    // HACK: we must delete it->second, but for some reasons, deleting this incurs some problem.
                    delete it->second;
#endif
                    it = train_task_map_.erase(it);
                } else {
                    ++it;
                }
            }
        }

        /* iterate with lock to find tasks to wake */
        {
            std::unique_lock<std::mutex> ul(train_task_map_mutex_);
            for (auto it = train_task_map_.begin(); it != train_task_map_.end();++it) {
                auto task = it->second;

                std::unique_lock<std::mutex> ul(task->mutex_);
                auto &barrier = task->shared_props_->train_barrier_[task->barrier_id_];
                pthread_mutex_lock(&barrier.mutex_);
                unsigned barrier_idx = task->barrier_id_;

                if (task->state() == TASK_PENDING_BARRIER && barrier.state_ == FINISHED) {
                    assert(barrier.count_ > 0);
                    barrier.count_--;
                    if (barrier.count_ == 0) {
                        barrier.state_ = UNINITIALIZED;
                    }
                    task->barrier_id_++;
                    tasks_to_wake.push_back(std::make_pair(task, barrier_idx));
                }
                pthread_mutex_unlock(&barrier.mutex_);
            }
        }

        /* waking up found threads (no lock) */
        {
            for(auto it = tasks_to_wake.begin(); it != tasks_to_wake.end(); ++it) {
                auto task = it->first;
                unsigned barrier_idx = it->second;
                if (task->barrier_callback_fn_[barrier_idx] != nullptr) {
                    task->state_update(TASK_PENDING);
                    (*task->barrier_callback_fn_[barrier_idx])();
                    task->barrier_callback_fn_[barrier_idx] = nullptr;
                }
            }
        }

        /**
         *  Note: there may be tasks to wake at this point, as we are not enforcing global lock for performance.
         *  However, in this case, we may experience intermittent indefinite locks, for example, when the signal arrives before issuing cond_wait.
         *  To mitigate this, we enforce wait with 100ms timeout here.
        */

        {
            const unsigned long delay_ms = 100;
            struct timespec waitUntil;
            struct timeval now;
            gettimeofday(&now, nullptr);
            waitUntil.tv_sec = now.tv_sec;
            waitUntil.tv_nsec = (now.tv_usec+1000UL*delay_ms)*1000UL;
            if (waitUntil.tv_nsec >= 1000000000UL) {
                waitUntil.tv_nsec -= 1000000000UL;
                waitUntil.tv_sec -= 1;
            }
            pthread_mutex_lock(&shared_props_->barrier_ipc_mutex_);
            pthread_cond_timedwait(&shared_props_->barrier_ipc_cond_, &shared_props_->barrier_ipc_mutex_, &waitUntil);
            pthread_mutex_unlock(&shared_props_->barrier_ipc_mutex_);
        }
    }

    return 0;
}


/**
 * Delegates post_backward_process 
*/
int FasterDpEngine::backward_delegate_thread_main() {
    sigset_t mask;
    sigemptyset (&mask);
    sigaddset (&mask, SIGTERM);
    sigaddset (&mask, SIGHUP);
    pthread_sigmask(SIG_BLOCK, &mask, NULL);
    pthread_setname_np(pthread_self(), "BwdDelegate");

    while (!finished_) {
        std::shared_ptr<std::packaged_task<void ()>> fn = nullptr;
        {
            std::unique_lock<std::mutex> ul(backward_delegate_mutex_);
            backward_delegate_cond_.wait(ul, [this] () { return list_backward_delegate_.size() > 0 || finished_; });

            if (list_backward_delegate_.size() == 0 && finished_)
                break;

            assert(list_backward_delegate_.size() > 0);

            fn = list_backward_delegate_.front();
            list_backward_delegate_.pop_front();
        }
        assert(fn);
        nvtxRangePush("post_backward_process_delegate");
        (*fn)();
        nvtxRangePop();
    }
    return 0;
}

static void printHexDump(const void* data, size_t size) {
    const unsigned char* p = (const unsigned char*)data;

    for (size_t i = 0; i < size; ++i) {
        printf("%02X ", p[i]);

        // Print an extra space after every 8 bytes
        if ((i + 1) % 8 == 0)
            printf(" ");

        // Print an extra newline after every 16 bytes
        if ((i + 1) % 16 == 0)
            printf("\n");
    }

    // Add a newline if the last line was not complete
    if (size % 16 != 0)
        printf("\n");
}


TrainTaskSharedProps *FasterDpEngine::retrieve_train_task_shared_props(const std::string key) {
    
    pthread_mutex_lock(&shared_props_->global_ipc_mutex_);
    const auto canary = shared_props_->canary_;
    shared_props_->canary_++;

    /* implement simple ring buffer */
    const off64_t head = shared_props_->task_buffer_head_; 
    const off64_t tail = shared_props_->task_buffer_tail_;
    auto *buf = shared_props_->task_buffer_;
    TrainTaskSharedProps *shared_ptr = nullptr;

    assert(shared_props_->task_alloc_num_ < NUM_MAX_CONCURRENT_TASKS);

    if (head < tail) {
        for (off64_t p = head; p < tail; p++) {
            if (buf[p].valid_ && strcmp(key.c_str(), buf[p].key_) == 0) {
                shared_ptr = &buf[p];
                // std::cout << "Found Existing Task " << std::endl;
                break;
            }
        }
    } else if (tail < head) {
        for (off64_t p = head; p < NUM_MAX_CONCURRENT_TASKS; p++) {
            if (buf[p].valid_ && strcmp(key.c_str(), buf[p].key_) == 0) {
                shared_ptr = &buf[p];
                // std::cout << "Found Existing Task " << std::endl;
                break;
            }
        }
         
        for (off64_t p = 0; p < head; p++) {
            if (buf[p].valid_ && strcmp(key.c_str(), buf[p].key_) == 0) {
                shared_ptr = &buf[p];
                // std::cout << "Found Existing Task " << std::endl;
                break;
            }
        }
    }

    if (shared_ptr == nullptr) {
        assert(tail == shared_props_->task_buffer_tail_);
        assert((tail + 1) % NUM_MAX_CONCURRENT_TASKS != head);
        shared_ptr = buf + tail;
        shared_props_->task_buffer_tail_ = (shared_props_->task_buffer_tail_ + 1) % NUM_MAX_CONCURRENT_TASKS;
        // std::stringstream s;
        // s << "Alloc Pos " << tail << ", set head=" << shared_props_->task_buffer_head_ << ", tail=" << shared_props_->task_buffer_tail_ << ", allocated by " << local_rank_;
        // std::cerr << s.str() << std::endl;

        if (shared_ptr->valid_) {
            std::cerr << "Error: Expected invalid chunks, got valid chunk. Allocated " << shared_props_->task_alloc_num_ << " tasks so far." << std::endl;
            std::cerr << "head=" << shared_props_->task_buffer_head_ << ", tail=" << shared_props_->task_buffer_tail_ << std::endl;
            printHexDump(shared_ptr, sizeof(TrainTaskSharedProps));
        }
        shared_ptr->initialize(key);
        shared_props_->task_alloc_num_++;
    }


    /* validity check */
    if (shared_props_->task_buffer_head_ <= shared_props_->task_buffer_tail_ && 
        shared_props_->task_alloc_num_ > shared_props_->task_buffer_tail_ - shared_props_->task_buffer_head_) {
        std::cerr << "[1] head=" << shared_props_->task_buffer_head_ 
                    << ", tail=" << shared_props_->task_buffer_tail_
                    << ", num=" << shared_props_->task_alloc_num_ << std::endl;
        abort();
    }
    if (shared_props_->task_buffer_head_ > shared_props_->task_buffer_tail_ && 
        shared_props_->task_alloc_num_ > shared_props_->task_buffer_tail_  + NUM_MAX_CONCURRENT_TASKS - shared_props_->task_buffer_head_) {
        std::cerr << "[2] head=" << shared_props_->task_buffer_head_ 
                    << ", tail=" << shared_props_->task_buffer_tail_
                    << ", num=" << shared_props_->task_alloc_num_ << std::endl;
        abort();
    }

    assert(shared_props_->task_buffer_head_ > shared_props_->task_buffer_tail_ || 
            shared_props_->task_buffer_tail_ - shared_props_->task_buffer_head_ >= shared_props_->task_alloc_num_);
    assert(shared_props_->task_buffer_head_ <= shared_props_->task_buffer_tail_ || 
            shared_props_->task_buffer_tail_  + NUM_MAX_CONCURRENT_TASKS - shared_props_->task_buffer_head_ >= shared_props_->task_alloc_num_);

    assert(shared_ptr);

    if(canary + 1 != shared_props_->canary_) {
        std::cerr << "Canary: Expected to be " << (canary + 1) << ", but shared canary = " <<  shared_props_->canary_ << std::endl;
    }
    pthread_mutex_unlock(&shared_props_->global_ipc_mutex_);

    return shared_ptr;
}
    
void FasterDpEngine::free_task(TrainTaskV2 *task) {
    bool initiate_shutdown = false;
    assert(task);
    assert(task->shared_props_);

    /* try to invalidate shared props */
    auto shprops = task->shared_props_;
    pthread_mutex_lock(&shprops->mutex_);
    shprops->is_finished_[local_rank_] = true;
    if (!shprops->finish_initiated_) {
        bool everything_is_finished = true;
        for (unsigned i = 0; i < node_world_size_; i++) {
            if (!shprops->is_finished_[i]) {
                everything_is_finished = false;
                break;
            }
        }

        if (everything_is_finished) {
            shprops->finish_initiated_ = true;
            initiate_shutdown = true;
        }
    }
    pthread_mutex_unlock(&task->shared_props_->mutex_);

    /* invalidate local task */
    // {
    //     std::stringstream s;
    //     s << "Invalidating local state for task=" << task->key();
    //     std::cerr << s.str() << std::endl;
    // }

    {
        std::unique_lock<std::mutex> ul(task->mutex_);
        task->invalidate();
    }

    /* invalidate shared props if this is the last process */    
    if (initiate_shutdown) {
        assert(shprops >= shared_props_->task_buffer_);
        assert(shprops < shared_props_->task_buffer_ + NUM_MAX_CONCURRENT_TASKS);
        assert(shprops->valid_);


        off64_t pos = shprops - shared_props_->task_buffer_;
        
        pthread_mutex_lock(&shared_props_->global_ipc_mutex_);
        shprops->valid_ = false;

        while (!shared_props_->task_buffer_[shared_props_->task_buffer_head_].valid_ && 
                shared_props_->task_buffer_head_ != shared_props_->task_buffer_tail_) {
            shared_props_->task_buffer_head_ = (shared_props_->task_buffer_head_ + 1) % NUM_MAX_CONCURRENT_TASKS;
        }
        assert(shared_props_->task_alloc_num_ > 0);
        shared_props_->task_alloc_num_--;
        const auto result = (shared_props_->task_buffer_head_);
        
        // std::stringstream s;
        // s << "Free  Pos " << pos << ", set head = " << result << ", tail = " << (shared_props_->task_buffer_tail_);
        // std::cerr << s.str() << std::endl;
        pthread_mutex_unlock(&shared_props_->global_ipc_mutex_);
    }
}

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

void FasterDpEngine::pre_forward_process(int layer_idx, std::string name) {
    // keep monitor and make sure that pre_forward call count == model version!
    {
        std::unique_lock<std::mutex> ul(layer_model_map_mutex_);
        const std::string skey = TrainTaskV2::to_persistent_key(layer_idx, name);
        if (layer_model_version_map_.find(skey) == layer_model_version_map_.end()) {
            // std::cerr << "New layer " << skey << std::endl;
            layer_model_version_map_[skey] = 0;
        }
        if (layer_iteration_cnt_map_.find(skey) == layer_iteration_cnt_map_.end()) {
            layer_iteration_cnt_map_[skey] = 0;
        } else {
            layer_iteration_cnt_map_[skey]++;
        }

        assert(layer_model_version_map_.find(skey) != layer_model_version_map_.end());
        assert(layer_iteration_cnt_map_.find(skey) != layer_iteration_cnt_map_.end());

        while (true) {
            const int model_version = layer_model_version_map_[skey];
            const int current_iteration = layer_iteration_cnt_map_[skey];

            if (model_version >= current_iteration - model_staleness_) {
                // std::cerr << "Layer " << layer_idx << " " << name << " has model version " << model_version << " at iter " << current_iteration << std::endl;
                break;
            }
            layer_model_map_cond_.wait(ul);
        }
        

        const int model_version = layer_model_version_map_[skey];
        const int current_iteration = layer_iteration_cnt_map_[skey];
        if (model_version < current_iteration - model_staleness_) {
            std::cerr << "Error: Layer " << layer_idx << " " << name << " has model version " << model_version << " at iter " << current_iteration << std::endl;
            abort();
        }
    }

    {
        const std::string skey = TrainTaskV2::to_persistent_key(layer_idx, name);
        std::unique_lock<std::mutex> ul(layer_model_completed_version_map_mutex_);
        if (layer_model_completed_version_map_.find(skey) == layer_model_version_map_.end()) {
            layer_model_completed_version_map_[skey] = -1;
        }
    }

}


void FasterDpEngine::pre_train_init(int layer_idx, std::string name, torch::Tensor gpu_param) {
    const std::string skey = TrainTaskV2::to_persistent_key(layer_idx, name);
   
    {
        std::unique_lock<std::mutex> ul(set_initialized_params_mutex_);
        if (set_initialized_params_.find(skey) != set_initialized_params_.end()) {
            /* already initialized */
            return;
        } else {
            set_initialized_params_.emplace(skey);
        }
    }

    /* register hook here */
    gpu_param.register_hook([this, layer_idx, name, gpu_param, skey] (torch::Tensor grad) {
        // if (skey == std::string("17@bias")) {
        //     std::cerr << "Grad norm = " << grad.norm().cpu().item() << ", at ptr = " << grad.data_ptr() << std::endl;
        // }
    
        /* need to accumulate grad somewhere. `grad` is not an accumulated gradient */
        {
            std::unique_lock<std::mutex> ul(map_gpu_grad_tensor_mutex_);
            if (map_gpu_grad_tensor_.find(skey) == map_gpu_grad_tensor_.end()) {
                map_gpu_grad_tensor_[skey] = grad.clone();
            } else {
                map_gpu_grad_tensor_[skey].add_(grad);
            }
        }
        
        // if (skey == std::string("17@bias")) {
        //     std::cerr << "Grad norm.acc = " <<  map_gpu_grad_tensor_[skey].norm().cpu().item() << ", at ptr = " << map_gpu_grad_tensor_[skey].data_ptr() << std::endl;
        // }

        post_backward_process(layer_idx, name, map_gpu_grad_tensor_[skey], gpu_param);

        // post_backward_process(layer_idx, name, grad, gpu_param);
    });


    if (!node_master_) {
        std::unique_lock<std::mutex> ul(map_gpu_param_tensor_mutex_);
        assert(map_gpu_param_tensor_.find(skey) == map_gpu_param_tensor_.end());
        map_gpu_param_tensor_[skey] = gpu_param;
        return;
    }

    {
        std::unique_lock<std::mutex> ul(map_cpu_param_tensor_mutex_);
        assert(map_cpu_param_tensor_.find(skey) == map_cpu_param_tensor_.end());
        map_cpu_param_tensor_[skey] = shm_manager_->alloc(gpu_param.sizes(), false);
    }
    {
        std::unique_lock<std::mutex> ul(map_gpu_param_tensor_mutex_);
        assert(map_gpu_param_tensor_.find(skey) == map_gpu_param_tensor_.end());
        map_gpu_param_tensor_[skey] = gpu_param;
    }

    auto &cpu_param_tensor = map_cpu_param_tensor_[skey];

    pthread_mutex_lock(&shared_props_->global_ipc_mutex_);
    auto &pos = shared_props_->cpu_param_lst_[shared_props_->cpu_param_num_];
    pos.offset = shm_manager_->get_offset(cpu_param_tensor.data_ptr());
    strncpy(pos.name, skey.c_str(), sizeof(pos.name));
    memcpy(pos.len, cpu_param_tensor.sizes().data(), sizeof(int64_t) * cpu_param_tensor.sizes().size());
    pos.len_len = cpu_param_tensor.sizes().size();
    shared_props_->cpu_param_num_++;
    assert(shared_props_->cpu_param_num_ < NUM_MAX_LAYERS);
    pthread_mutex_unlock(&shared_props_->global_ipc_mutex_);

    cpu_param_tensor.copy_(gpu_param);
    cudaDeviceSynchronize();

    force_model_sync(skey);
}

void FasterDpEngine::force_model_sync(int layer_idx, std::string name, bool dry_run) {
    std::string skey = TrainTaskV2::to_persistent_key(layer_idx, name);
    force_model_sync(skey, dry_run);
}

void FasterDpEngine::force_model_sync(std::string skey, bool dry_run) {

#if not DEBUG_ACCURACY
    if (dry_run) {
        return;
    }
#endif

    if (!node_master_) {
        return;
    }

    if (world_size_ == 1) {
        return;
    }

    // std::cerr << "Performing model sync for " << skey << std::endl;
    auto &cpu_param_tensor = get_cpu_param_tensor(skey);
    if (master_) {
        comm_manager_->sendInitmodel(skey, reinterpret_cast<float *>(cpu_param_tensor.data_ptr()), cpu_param_tensor.numel());
    } else {
        {
            auto spec = comm_manager_->recvInitmodel(skey);
            // std::cerr << "Received model for " << spec.skey << std::endl;
            init_model_recv_map_[spec.skey] = std::move(spec);
        }

        for (auto it = init_model_recv_map_.begin(); it != init_model_recv_map_.end();) {
            std::string skey = it->second.skey;
            if (map_cpu_param_tensor_.find(skey) != map_cpu_param_tensor_.end()) {
                auto &dst = map_cpu_param_tensor_[skey];
                // std::cerr << "Applied model for " << skey << ", dst.numel=" << dst.numel() << " mdllen=" << it->second.len << std::endl;
                assert(dst.numel() == it->second.len);
                auto src = torch::from_blob(it->second.ptr.get(), dst.sizes());
                auto norm_diff = (dst - src).norm() / src.norm();
                // std::cerr << "Performing model sync for " << skey << " OK, norm: " << src.norm().item<float>() << " -> " << dst.norm().item<float>() << ", error rate = " << norm_diff.item<float>()*100. << "%" << std::endl;
                if (!dry_run)
                    map_cpu_param_tensor_[skey].copy_(src);
                it = init_model_recv_map_.erase(it);
            } else {
                ++it;
            }
        }
    }
}


unsigned FasterDpEngine::update_layer_model_version(const std::string skey) {
    std::unique_lock<std::mutex> ul(layer_alloc_cnt_map_mutex_);
    const auto it = layer_alloc_cnt_map_.find(skey);
    if (it == layer_alloc_cnt_map_.end())
        layer_alloc_cnt_map_[skey] = 0;
    unsigned iter_count = layer_alloc_cnt_map_[skey];
    layer_alloc_cnt_map_[skey]++;
    return iter_count;
}

void FasterDpEngine::notify_layer_usage_finished(const std::string skey, unsigned iter_idx) {
    {
        std::unique_lock<std::mutex> ul(layer_model_completed_version_map_mutex_);
        const auto it = layer_model_completed_version_map_.find(skey);
        assert(it != layer_model_completed_version_map_.end());
        layer_model_completed_version_map_[skey] = std::max(-1, (int)iter_idx - model_staleness_);
        
        // std::cerr << "Updating complete version of " << skey << " to " << layer_model_completed_version_map_[skey] << " at " << iter_count << std::endl;
    }
    layer_model_completed_version_map_cond_.notify_all();
}

void FasterDpEngine::notify_all_layer_usage_finished() {
    std::unique_lock<std::mutex> ul(layer_alloc_cnt_map_mutex_);
    for (auto it = layer_alloc_cnt_map_.begin(); it != layer_alloc_cnt_map_.end(); ++it) {
        const auto &skey = it->first;
        const auto iter_idx = it->second;
        
        std::unique_lock<std::mutex> ul2(layer_model_completed_version_map_mutex_);
        const auto it2 = layer_model_completed_version_map_.find(skey);
        assert(it2 != layer_model_completed_version_map_.end());
        layer_model_completed_version_map_[skey] = std::max(-1, (int)iter_idx - model_staleness_);
    }
    layer_model_completed_version_map_cond_.notify_all();
}

int FasterDpEngine::update_micro_iteration(const std::string skey) {
    std::unique_lock<std::mutex> ul(layer_uiter_map_mutex_);
    const auto it = layer_uiter_map_.find(skey);
    if (it == layer_uiter_map_.end())
        layer_uiter_map_[skey] = 0;
    int uiter_idx = layer_uiter_map_[skey];
    layer_uiter_map_[skey] = (layer_uiter_map_[skey] + 1) % gradient_accumulation_;
    return uiter_idx;
}

torch::Tensor &FasterDpEngine::get_cpu_param_tensor(std::string skey) { 

    std::unique_lock<std::mutex> ul(map_cpu_param_tensor_mutex_);
    auto it = map_cpu_param_tensor_.find(skey);
    if (it != map_cpu_param_tensor_.end())
        return it->second;


    // find from global map
    pthread_mutex_lock(&shared_props_->global_ipc_mutex_);
    for(int i = 0; i < shared_props_->cpu_param_num_; i++) {
        auto &itm = shared_props_->cpu_param_lst_[i];
        if (strcmp(itm.name, skey.c_str()) == 0) {

            map_cpu_param_tensor_[itm.name] = shm_manager_->tensor_from(
                itm.offset, c10::IntArrayRef(itm.len, itm.len_len)
            );
            
            // std::cerr << "Recreating cpu_param_tensor " << skey << " of size " << map_cpu_param_tensor_[itm.name].sizes() << std::endl;
            pthread_mutex_unlock(&shared_props_->global_ipc_mutex_);
            return map_cpu_param_tensor_[itm.name];
        }
    }
    pthread_mutex_unlock(&shared_props_->global_ipc_mutex_);

    assert(false);
    return map_cpu_param_tensor_.begin()->second;
}

SharedCpuTensorEntry *FasterDpEngine::retrieve_cpu_shared_tensor_entry(int iter_idx, const std::string skey, c10::IntArrayRef tensor_shape) {
    /* find in local context */
    
    std::unique_lock<std::mutex> ul(shared_cpu_tensor_entry_map_mutex_);
    const int dup_iter_idx = iter_idx % (model_staleness_ + 1);
    auto spkey = std::make_pair(skey, dup_iter_idx);

    auto local_it = shared_cpu_tensor_entry_map_.find(spkey);
    if (local_it != shared_cpu_tensor_entry_map_.end()) {
        auto entry = local_it->second;
        return entry;
    }

    /* find in global context */
    pthread_mutex_lock(&shared_props_->global_ipc_mutex_);

    SharedCpuTensorEntry *entry = nullptr;
    for (uint32_t i = 0; i < shared_props_->cpu_tensor_num_; i++) {
        auto e = &shared_props_->cpu_tensor_lst_[i];
        if (e->valid_ && strcmp(e->persistent_key_, skey.c_str()) == 0 && e->entry_dup_idx_ == dup_iter_idx) {
            entry = e;
            break;
        }
    }

    /* if there is no entry, make one */
    if (entry == nullptr) {
        entry = &shared_props_->cpu_tensor_lst_[shared_props_->cpu_tensor_num_];
        entry->initialize(skey, dup_iter_idx);
        shared_props_->cpu_tensor_num_++;
        // std::cerr << "Allocating new cpu_shared_tensor_entry : " << skey << ", shared_entry=" << entry << std::endl;
    }
    pthread_mutex_unlock(&shared_props_->global_ipc_mutex_);

    /* fill CPU param and grad tensors of this process */
    if (local_rank_ == 0) {
        assert(map_cpu_param_tensor_.find(skey) != map_cpu_param_tensor_.end());
        auto &param_tensor = map_cpu_param_tensor_[skey];
        
        auto &param_tensor_mock = entry->param_;
        param_tensor_mock.valid = true;
        param_tensor_mock.offset = shm_manager_->get_offset(param_tensor.data_ptr());
        memcpy(param_tensor_mock.len, param_tensor.sizes().data(), sizeof(int64_t) * param_tensor.sizes().size());
        param_tensor_mock.len_len = param_tensor.sizes().size();
    }
    
    auto grad_tensor = shm_manager_->alloc(tensor_shape, false, true);
    auto &grad_tensor_mock = entry->grad_[local_rank_];
    grad_tensor_mock.valid = true;
    grad_tensor_mock.offset = shm_manager_->get_offset(grad_tensor.data_ptr());
    memcpy(grad_tensor_mock.len, grad_tensor.sizes().data(), sizeof(int64_t) * grad_tensor.sizes().size());
    grad_tensor_mock.len_len = grad_tensor.sizes().size();

    shared_cpu_tensor_entry_map_[spkey] = entry;
    // std::cerr << "Assigning " << entry << " to " << skey << " at iter " << iter_idx << std::endl;

    assert(entry);
    assert(entry->valid_);
    return entry;
}


void FasterDpEngine::post_backward_process(int layer_idx, std::string name, torch::Tensor gpu_grad_tensor, torch::Tensor gpu_param_tensor) {

    // std::cerr << "post_backward_process : " << layer_idx << ", name = " << name << std::endl;
    nvtxRangePush("post_backward_process");

    /* synchronize across intra-node before the first backward step */
    if (first_backward_) {
        first_backward_ = false;
        barrier();
    }

    std::unique_lock<std::mutex> ul(backward_delegate_mutex_);
    list_backward_delegate_.emplace_back
    (std::make_shared<std::packaged_task<void()>>(std::bind(std::forward<std::function<void()>>(
        [this, layer_idx, name, gpu_grad_tensor, gpu_param_tensor] () {
            
            const std::string skey = TrainTaskV2::to_persistent_key(layer_idx, name);

            const int uiter_idx = update_micro_iteration(skey);
            if (uiter_idx < gradient_accumulation_ - 1) {
                /* Skipping backward, will accumulate gradient */
                return;
            }
            unsigned iter_idx = update_layer_model_version(skey);
            notify_layer_usage_finished(skey, iter_idx);

            
            auto task = new TrainTaskV2(iter_idx, layer_idx, name);
            const std::string key = task->key();

            // std::cerr << "post_backward_process: " << key << " : grad_tensor=" << gpu_grad_tensor.data_ptr() << ", param_tensor=" << gpu_param_tensor.data_ptr() << std::endl;

            /* configure tasks */
            nvtxRangePush("gpu_param_check");
            {
                auto saved_gpu_param_tensor = get_gpu_param_tensor(skey);
                if (saved_gpu_param_tensor.data_ptr() != gpu_param_tensor.data_ptr()) {
                    throw std::runtime_error("GPU parameter tensor does not match!");
                }
            }
            nvtxRangePop();

            nvtxRangePush("create_task");
            task->assign_gpu_tensor(gpu_param_tensor, gpu_grad_tensor); /* later need to check if gpu_param_tensor = map_gpu_param_tensor_[skey] */
            task->tensor_numel_ = gpu_grad_tensor.numel();
            task->shared_cpu_tensor_entry_ = retrieve_cpu_shared_tensor_entry(iter_idx, skey, gpu_grad_tensor.sizes());
            task->shared_props_ = retrieve_train_task_shared_props(key);
            nvtxRangePop();

            /* register to train_task_map_ here to prevent potential race */
            {
                nvtxRangePush("train_task_map_mutex_");
                std::unique_lock<std::mutex> ul(train_task_map_mutex_);
                train_task_map_[key] = task;
                train_task_set_.insert(task);
                nvtxRangePop();
            }
            
            nvtxRangePush("schedule_after_use");
#if DEBUG_BARRIER
            schedule(module_barrier_checker_, task);
#else
            schedule_after_use(module_d2h_copy_, task);
#endif
            nvtxRangePop();
        }))));

    backward_delegate_cond_.notify_one();

    nvtxRangePop();
}

void FasterDpEngine::synchronize_backend() {

    nvtxRangePush("synchronize_backend");

    /* Notify that all backward layer usage is finished */
    notify_all_layer_usage_finished();

    std::unique_ptr<std::list<std::future<void>>> lst_futures_to_process;

    {
        std::unique_lock<std::mutex> ul2(lst_futures_write_mutex_); // stop writing to lst_futures_!
        lst_futures_to_process = std::move(lst_futures_);
        lst_futures_ = std::make_unique<std::list<std::future<void>>>();
    }
    
    /* Allow other threads to add to new lst_futures_ */

    assert(lst_futures_to_process != nullptr);
    std::cerr << "Synchronize Backend, num_futures: " << lst_futures_to_process->size() << std::endl;


    size_t cnt = 0;
    for(auto &fut : *lst_futures_to_process) {
        if (fut.wait_for(std::chrono::seconds(0)) != std::future_status::ready) {
            // std::cerr << "Waiting for future" << std::endl;
        }
        // fut.wait();
        while (fut.wait_for(std::chrono::seconds(5)) != std::future_status::ready) {
            std::cerr << "Waiting for future for more than 5 sec, idx=" << cnt << std::endl;
            std::cerr << "DBG STAT = " << 
                (fut.wait_for(std::chrono::seconds(0)) == std::future_status::timeout ? "timeout" : "deferred") << std::endl;
            pthread_cond_broadcast(&shared_props_->barrier_ipc_cond_);

#if ENABLE_STAT
            stat_export();
#endif
        }
        cnt++;
    }

    lst_futures_to_process->clear();
    
    nvtxRangePop();
}


torch::Tensor &FasterDpEngine::get_gpu_param_tensor(std::string skey) {
    std::unique_lock<std::mutex> ul(map_gpu_param_tensor_mutex_);
    auto it = map_gpu_param_tensor_.find(skey);
    assert(it != map_gpu_param_tensor_.end());
    return it->second;
}

#if ENABLE_STAT
void FasterDpEngine::record_stat_impl(TrainTaskV2 *task, const std::string &event, bool is_end) {
    const std::chrono::time_point<std::chrono::steady_clock> now = std::chrono::steady_clock::now();

    {
        std::unique_lock<std::mutex> ul(train_task_map_mutex_);
        if (train_task_set_.find(task) == train_task_set_.end()) {
            std::cerr << "Illegal write" << std::endl;
            return;
        }
    }
    const std::string key(task->key());
    std::unique_lock<std::mutex> gl(task_stat_map_mutex_);
    task_stat_last_event_ = key;
    auto & stat_item = task_stat_map_[key];
    gl.unlock();
    
    std::unique_lock<std::mutex> ul(stat_item.mutex);
    if (is_end) {
        stat_item.entries[event].end = now;
    } else {
        stat_item.entries[event].begin = now;
    }
}


void FasterDpEngine::stat_export() {
    using json = nlohmann::json;
    json j;

    std::unique_lock<std::mutex> gl(task_stat_map_mutex_);    

    for (auto it = task_stat_map_.begin(); it != task_stat_map_.end(); ++it) {
        const auto & task_key = it->first;
        auto & jtask = j[task_key] = {};

        for (auto it2 = it->second.entries.begin(); it2 != it->second.entries.end(); ++it2) {
            const auto & event_name = it2->first;
            const auto & event_stat = it2->second;
            double duration = std::chrono::duration_cast<std::chrono::microseconds>(event_stat.end - event_stat.begin).count() / 1000.;
            jtask[event_name] = event_stat.end.time_since_epoch().count() == 0 ? -1 : duration;
        }
        
    }

    std::stringstream fn_stream;
    auto now = std::chrono::system_clock::now();
    auto tt = std::chrono::system_clock::to_time_t(now);
    fn_stream << "/tmp/time_log_" << std::put_time(localtime(&tt), "%Y%m%d%H%M%S");
    fn_stream << "_" << getpid() << ".json";

    std::string fn = fn_stream.str();
    std::cerr << "Writing statistics at " << fn << ", last event was " << task_stat_last_event_ << std::endl;
    std::ofstream of(fn, std::ios_base::out);
    of << j;
    of.close();

}
#endif

std::pair<torch::Tensor, torch::Tensor> FasterDpEngine::compress(const std::string &name, const torch::Tensor &tensor, float ratio) {

    if (ratio < 0 || ratio > 1) {
        throw std::runtime_error("Ratio must be in range [0, 1].");
    }

    long numel_to_select = (1. - ratio) * tensor.numel();

    auto dst_idx = torch::empty({numel_to_select}, torch::TensorOptions().dtype(torch::kInt32));
    auto dst_val = torch::empty({numel_to_select});

    std::pair<const float *, const long unsigned int> src = std::make_pair(reinterpret_cast<const float *>(tensor.data_ptr()), tensor.numel());

    if (numel_to_select == 0) {
        return std::make_pair(dst_idx, dst_val);
    }


    auto total_len = compressor_->compress(
        name,
        src,
        numel_to_select,
        std::make_pair<uint32_t *, size_t>(reinterpret_cast<uint32_t *>(dst_idx.data_ptr()), numel_to_select),
        std::make_pair<float *, size_t>(reinterpret_cast<float *>(dst_val.data_ptr()), numel_to_select)
    );

    if (total_len != numel_to_select) {
        dst_idx = dst_idx.narrow(0, 0, total_len);
        dst_val = dst_val.narrow(0, 0, total_len);

        TORCH_CHECK(dst_idx.numel() == total_len);
        TORCH_CHECK(dst_val.numel() == total_len);
    }

    return std::make_pair(dst_idx, dst_val);
}


void FasterDpEngine::print_current_stat() const {

    const char * barrier_state_map[] = {
        "UNINITIALIZED",
        "INITIALIZED",
        "FINISHED"
    };

    std::stringstream s;
    for(auto it = train_task_map_.begin(); it != train_task_map_.end(); ++it) {
        auto task = it->second;
        auto &barrier = task->shared_props_->train_barrier_[task->barrier_id_];
        s << "Task [" << task;
        s << "]: " << task->key() << ":";
        s << "  s = " << train_task_state_name[task->state()];
        if (!task->valid()) {
            s << " [INVALID]";
        }
        s << " b = (" << task->barrier_id_ << "," << task->barrier_id_future_ << ") " << barrier_state_map[barrier.state_] << " " << barrier.name_ << " [cnt=" << barrier.count_ << "]";
        s << task->debug_log_.str();
        s << "\n";
    }
    std::cerr << s.str() << std::endl;
}


void FasterDpEngine::gdb_force_unblock() const {    
    pthread_cond_broadcast(&shared_props_->barrier_ipc_cond_);
    std::cerr << "Broadcast COND!" << std::endl;
}