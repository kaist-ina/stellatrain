#ifndef _ENGINE_TASK_H
#define _ENGINE_TASK_H
#include <stdint.h>
#include <stddef.h>
#include <pthread.h>
#include <string>
#include <future>
#include <chrono>
#include <torch/torch.h>
#include "config.h"

extern thread_local int worker_id;
extern int num_workers;


enum custom_barrier_state_t { 
    UNINITIALIZED = 0, 
    INITIALIZED = 1, 
    FINISHED = 2
};

struct TrainBarrier {
    char name_[MAX_BARRIER_NAME_LEN+1];
    custom_barrier_state_t state_;
    int count_;
    pthread_mutex_t mutex_;
};

/**
 * Struct to reconstruct CPU tensor from shared memory.
*/
class SharedCpuTensorMock {
public:
    bool valid;
    off64_t offset; /* data offset from the shared memory payload start ptr */
    int64_t len[NUM_MAX_PARAM_DIM]; /* tensor dimensions */
    uint32_t len_len; /* tensor dimension length */
};

static_assert(std::is_pod<SharedCpuTensorMock>::value, "Must be a POD type.");

/*
 * Defines Tensor Entry that resides in CPU shared memory.
 * Must be unique per (trainable) parameter tensor.
*/
class SharedCpuTensorEntry {
public:
    bool valid_;
    int entry_dup_idx_; /* iter_idx % (staleness + 1) */
    char persistent_key_[MAX_TASK_NAME_LEN+1];

    int param_version_;
    int grad_version_[MAX_NUM_GPU_PER_NODE];
    SharedCpuTensorMock param_;
    SharedCpuTensorMock grad_[MAX_NUM_GPU_PER_NODE];

    SharedCpuTensorEntry() = default;
    void initialize(const std::string &skey, int dup_iter_idx);
};
static_assert(std::is_pod<SharedCpuTensorEntry>::value, "Must be a POD type.");


/**
 * Defines shared state of task that resides in CPU shared memory.
 * Must be unique per task.
*/
class TrainTaskSharedProps {
public:
    bool valid_;
    char key_[MAX_TASK_NAME_LEN+1];

    pthread_mutex_t mutex_; /* Use pthread_mutex_t to make class POD */

    /* used only for verification purposes */
    bool shared_cpu_data_ready_[MAX_NUM_GPU_PER_NODE];

    /* task termination related */
    volatile bool finish_initiated_;
    bool is_finished_[MAX_NUM_GPU_PER_NODE];

    int shared_test_field_[MAX_NUM_GPU_PER_NODE];
    /* barrier to synchronize between intra-node processes */
    TrainBarrier train_barrier_[MAX_NUM_BARRIER];

    TrainTaskSharedProps() = default;

    /* Any process can start initialization for TrainTaskSharedProps. However, each TrainTaskSharedProps must be initialized only once. */
    void initialize(const std::string &key);
    void lock();
    void unlock();
};

static_assert(std::is_pod<TrainTaskSharedProps>::value, "Must be a POD type.");


enum class TrainTaskState {

};

enum train_task_state_t {
    TASK_INITIALIZED = 0,
    TASK_PENDING = 1,
    TASK_RUNNING = 2,
    TASK_PENDING_CUDA = 3,
    TASK_PENDING_BARRIER = 4,
    TASK_PENDING_COMM = 5,
    TASK_PENDING_MODEL_COMPLETE = 6,
    TASK_PENDING_USE = 7,
    TASK_FINISHED = 8,
    TASK_IDLE = 9,
};

class FasterDpEngine;

class TrainTaskV2 {
private:
    static FasterDpEngine * engine_;
    const std::string key_;
    const std::string persistent_key_;
    torch::Tensor gpu_param_tensor_;
    torch::Tensor gpu_grad_tensor_;
    train_task_state_t state_;
    std::vector<std::pair<train_task_state_t, train_task_state_t>> state_history_;
    bool valid_;
    int iter_;
    int priority_;
    
    void state_update_impl(const train_task_state_t desired_state);
    
public:
    static void set_engine(FasterDpEngine *engine);
    static std::string to_key(unsigned iter_cnt, int layer, const std::string &key_str);
    static std::string to_persistent_key(int layer, const std::string &key_str);
    
    std::shared_ptr<std::packaged_task<void ()>> barrier_callback_fn_[MAX_NUM_BARRIER];
    unsigned barrier_id_; /* head: unclaimed barrier_id that was used in the past */
    unsigned barrier_id_future_; /* tail: barrier_id to use in future */
    std::mutex mutex_;
    std::stringstream debug_log_;
    std::string debug_msg_;

    TrainTaskSharedProps *shared_props_;
    SharedCpuTensorEntry *shared_cpu_tensor_entry_;

    unsigned grad_sync_iter_;
    float *compressed_grad_val_ptr_;
    uint32_t *compressed_grad_idx_ptr_;
    size_t tensor_numel_;
    int64_t tensor_compressed_numel_;
    
    std::chrono::steady_clock::time_point grad_sync_start_;

    int test_field_;

    inline const int &iter() const { return iter_; } 
    inline const std::string &key() const { return key_; } 
    inline const std::string &persistent_key() const { return persistent_key_; }
    inline const train_task_state_t state() const { return state_; }
    inline const bool valid() const { return valid_ == true; }
    inline void invalidate() { valid_ = false; }
    inline torch::Tensor &gpu_param_tensor() { return gpu_param_tensor_; }
    inline torch::Tensor &gpu_grad_tensor() { return gpu_grad_tensor_; }
    inline unsigned grad_sync_iter() const { return grad_sync_iter_; }
    inline const int priority() const { return priority_; }

    inline const std::vector<std::pair<train_task_state_t, train_task_state_t>> &state_history() { 
        assert(valid_); 
        return state_history_; 
    }

    TrainTaskV2(unsigned iter_cnt, int layer, const std::string key_str);

    void assign_gpu_tensor(const torch::Tensor &param_tensor, const torch::Tensor &grad_tensor);
    void free_gpu_grad_tensor();
    void free_gpu_param_tensor();
    void state_update(const train_task_state_t desired_state, bool apply_lock = true);
    void set_debug_message(const std::string msg) {
        debug_msg_ = msg;
    }
    inline const std::string &get_debug_message() { return debug_msg_; }

    void prepare_delete();
    
};


#endif