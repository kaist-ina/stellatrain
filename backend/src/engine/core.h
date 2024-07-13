#ifndef ENGINE_CORE_H
#define ENGINE_CORE_H

#include "config.h"
#include "logger.h"
#include "task.h"
#include "../compress/compressor.h"
#include "message.h"
#include <torch/torch.h>
#include <torch/extension.h>
#include <memory>
#include <list>
#include <future>
#include <utility>
#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <string>
#include <mutex>
#include <condition_variable>
#include <cassert>
#include <thread>
#include <functional>
#include <string>
#include <chrono>
#include <c10/cuda/CUDAStream.h>


class Module;
class ThreadPool;
class ShmManager;
class CommManager;
class SparseOptimizer;
struct SharedCoreProps;


struct StatEntry {
    std::chrono::time_point<std::chrono::steady_clock> begin;
    std::chrono::time_point<std::chrono::steady_clock> end;
};

struct StatItem {
    std::unordered_map<std::string, StatEntry> entries;
    std::mutex mutex;
};

class FasterDpEngine {
private:
    std::unique_ptr<ThreadPool> thread_pool_;
    std::unique_ptr<Compressor> compressor_;
    std::unique_ptr<ShmManager> shm_manager_;
    std::unique_ptr<CommManager> comm_manager_;
    std::unique_ptr<SparseOptimizer> sparse_optimizer_;
    std::unique_ptr<std::thread> barrier_manager_thread_;
    std::unique_ptr<std::thread> chore_manager_thread_;
    std::unique_ptr<std::thread> model_complete_manager_thread_;
    std::unique_ptr<std::thread> cpu_shmem_return_manager_thread_;
    std::unique_ptr<std::thread> backward_delegate_thread_;
    std::unique_ptr<std::list<std::future<void>>> lst_futures_;
    std::unordered_map<std::string, torch::Tensor> map_cpu_param_tensor_;
    std::unordered_map<std::string, torch::Tensor> map_gpu_param_tensor_;
    std::unordered_map<std::string, torch::Tensor> map_gpu_grad_tensor_;
    std::unordered_map<std::string, std::unique_ptr<float []>> map_compensate_grad_;
    mutable std::mutex map_cpu_param_tensor_mutex_;
    mutable std::mutex map_gpu_param_tensor_mutex_;
    mutable std::mutex map_gpu_grad_tensor_mutex_;
    mutable std::mutex lst_futures_write_mutex_;
    mutable std::mutex map_compensate_grad_mutex_;

    std::list<std::shared_ptr<std::packaged_task<void ()>>> list_backward_delegate_;
    mutable std::mutex backward_delegate_mutex_;
    std::condition_variable backward_delegate_cond_;

    std::unordered_set<std::string> set_initialized_params_;
    mutable std::mutex set_initialized_params_mutex_;
    

    std::string master_addr_;
    uint16_t master_port_;

    bool ready_;
    bool finished_;
    bool master_;
    bool node_master_;
    int local_session_id_;
    int rank_;
    int local_rank_;
    int world_size_;
    int node_world_size_;
    int gradient_accumulation_;

    int model_staleness_;

    bool first_backward_;

    double compression_ratio_;

    SharedCoreProps *shared_props_;

    std::condition_variable	barrier_cv_;
    std::shared_timed_mutex	barrier_mutex_;
    // std::unordered_map<uintptr_t, std::function<void()>> barrier_wait_map_;

    std::map<std::string, TrainTaskV2 *> train_task_map_;
    std::set<TrainTaskV2 *> train_task_set_;
    mutable std::mutex train_task_map_mutex_;

    std::map<std::pair<std::string, int>, SharedCpuTensorEntry *> shared_cpu_tensor_entry_map_; /* Shared CPU Tensor entry map indexed by (skey, iter_idx % (staleness+1))*/
    mutable std::mutex shared_cpu_tensor_entry_map_mutex_;

    std::unordered_map<std::string, unsigned> layer_alloc_cnt_map_;
    mutable std::mutex layer_alloc_cnt_map_mutex_;

    std::unordered_map<std::string, int> layer_uiter_map_;
    mutable std::mutex layer_uiter_map_mutex_;


    std::unordered_map<TrainTaskV2 *, std::shared_ptr<std::packaged_task<void ()>>> cuda_wait_callback_map_;
    mutable std::mutex cuda_wait_callback_map_mutex_;

    std::unordered_map<std::string, int> layer_iteration_cnt_map_;
    std::unordered_map<std::string, int> layer_model_version_map_;
    std::unordered_map<std::string, int> layer_model_completed_version_map_;
    mutable std::mutex layer_model_map_mutex_;
    mutable std::mutex layer_model_completed_version_map_mutex_;
    std::condition_variable layer_model_map_cond_;
    std::condition_variable layer_model_completed_version_map_cond_;
    std::unordered_map<TrainTaskV2 *, std::shared_ptr<std::packaged_task<void ()>>> layer_model_completed_callback_map_;

    std::unordered_map<std::string, RecvModelSpec> init_model_recv_map_;

    std::map<std::pair<std::string, int>, TrainTaskV2 *> cpu_shmem_use_map_;
    std::unordered_map<TrainTaskV2 *, std::shared_ptr<std::packaged_task<void ()>>> cpu_shmem_use_callback_map_;
    mutable std::mutex cpu_shmem_use_map_mutex_;
    std::condition_variable cpu_shmem_use_map_cond_;

    int barrier_manager_thread_main();
    int chore_manager_thread_main();
    int model_complete_manager_thread_main();
    int cpu_shmem_return_manager_thread_main();
    int backward_delegate_thread_main();

    std::condition_variable finished_cond_;

    FasterDpEngine ();
    static FasterDpEngine* instance;

    void schedule_after_use_impl(TrainTaskV2 *task, std::function<void()> callback_fn);
    void schedule_after_barrier_impl(TrainTaskV2 *task, std::function<void()> callback_fn, std::string barrier_name);
    void schedule_after_cuda_impl(TrainTaskV2 *task, std::function<void()> callback_fn);
    void schedule_after_comm_impl(TrainTaskV2 *task, float * dst_val, uint32_t * dst_idx, size_t len, std::function<void()> callback_fn, unsigned iter);
    void schedule_after_model_complete_impl(TrainTaskV2 *task, std::function<void()> callback_fn);


    void load_modules();
    
    void free_task(TrainTaskV2 *task);

    std::map<std::string, StatItem> task_stat_map_;
    mutable std::mutex task_stat_map_mutex_;
    std::string task_stat_last_event_;

#if ENABLE_STAT
    void record_stat_impl(TrainTaskV2 *task, const std::string &event, bool is_end);
    void stat_export();
#endif
    
    friend class TrainTaskV2;

public:
    // modules
    std::unique_ptr<Module> module_d2h_copy_;
    std::unique_ptr<Module> module_d2h_copy_post_;
    std::unique_ptr<Module> module_cpu_gather_;
    std::unique_ptr<Module> module_compress_;
    std::unique_ptr<Module> module_grad_exchange_;
    std::unique_ptr<Module> module_cpu_optimize_;
    std::unique_ptr<Module> module_h2d_copy_pre_;
    std::unique_ptr<Module> module_h2d_copy_;
    std::unique_ptr<Module> module_h2d_copy_post_;
#if DEBUG_BARRIER
    std::unique_ptr<Module> module_barrier_checker_;
#endif
    
    FasterDpEngine(FasterDpEngine const&) = delete;
    void operator=(FasterDpEngine const&) = delete;
    ~FasterDpEngine ();
    
    static FasterDpEngine& getInstance() {
        static FasterDpEngine instance;
        return instance;
    }

    void configure(std::string master_addr, uint16_t master_port, int world_size, int rank, int local_session_id, int local_world_size=0, int local_rank=0, const std::string method="", int gradient_accumulation=1);
    void configure_compression(const std::string &method);
    void configure_compression_ratio(double ratio);
    void pre_train_init(int layer_idx, std::string name, torch::Tensor gpu_param);
    void pre_forward_process(int layer_idx, std::string name);
    void post_backward_process(int layer_idx, std::string name, torch::Tensor gpu_grad_tensor, torch::Tensor gpu_param_tensor);
    void synchronize_backend();
    void barrier();
    void get_sparse_optimizer(const std::string &optimizer);
    bool is_ready() const { return ready_; }
    inline bool is_master() const { return master_; }
    inline bool is_node_master() const { return node_master_; }
    inline int local_rank() const { return local_rank_; }
    inline int node_world_size() const { return node_world_size_; }
    inline const std::unique_ptr<ShmManager> & shm_manager() const { return shm_manager_; }
    inline const std::unique_ptr<ThreadPool> & thread_pool() const { return thread_pool_; }
    inline const std::unique_ptr<Compressor> & compressor() const { return compressor_; }
    inline const std::unique_ptr<CommManager> & comm_manager() const { return comm_manager_; }
    inline const std::unique_ptr<SparseOptimizer> & sparse_optimizer() const { return sparse_optimizer_; }
    inline int n_threads() const { return thread_pool_->n_threads(); }
    inline int world_size() const { return world_size_; }
    inline double compression_ratio() const { return compression_ratio_; }
    inline int total_world_size() const { return world_size_ * node_world_size_; }
    inline int total_rank() const { return rank_ * node_world_size_ + local_rank_; }
    inline int gradient_accumulation() const { return gradient_accumulation_; }
    inline bool is_debug_accuracy_mode() const {
        return !!DEBUG_ACCURACY;
    }

    void schedule(std::unique_ptr<Module> & mod, TrainTaskV2 *task);
    void schedule_after_barrier(std::unique_ptr<Module> &mod, TrainTaskV2 *task, std::string barrier_name);
    void schedule_after_cuda(std::unique_ptr<Module> & mod, TrainTaskV2 *task);
    void schedule_after_model_complete(std::unique_ptr<Module> & mod, TrainTaskV2 *task);
    void schedule_after_comm(std::unique_ptr<Module> & mod, TrainTaskV2 *task, float * dst_val, uint32_t * dst_idx, size_t len, unsigned iter);
    
    void schedule_after_use(std::unique_ptr<Module> & mod, TrainTaskV2 *task); // automatically claimed
    void schedule_terminate(TrainTaskV2 *task);
    void return_cpu_shmem_after_use(TrainTaskV2 *task);
    
    static void CUDART_CB report_cuda_finished(void *userData);
    void report_cuda_finished_impl(TrainTaskV2 *task);

    std::pair<torch::Tensor, torch::Tensor> compress(const std::string &name, const torch::Tensor &tensor, float ratio);

    std::unique_ptr<float []> &get_grad_residual(TrainTaskV2 *task);
    void update_model_version(TrainTaskV2 *task);
    int get_model_version(TrainTaskV2 *task);
    int get_completed_model_version(TrainTaskV2 *task);

    TrainTaskV2 *find_task_by_key(const std::string key);
    void task_state_update_by_key(const std::string key, const train_task_state_t desired_state);
    void record_stat_start(TrainTaskV2 *task, const std::string event);
    void record_stat_end(TrainTaskV2 *task, const std::string event);

    void force_model_sync(std::string skey, bool dry_run = false);
    void force_model_sync(int layer_idx, std::string name, bool dry_run = false);

    void print_current_stat() const;
    void gdb_force_unblock() const;

    torch::Tensor tensor_from_mock(SharedCpuTensorMock &mock);

private:
    /** 
     * Increments the model version of the given layer.
     * @param skey persistent key.
     * @returns current model version before update (i.e. iteration index)
    */
    unsigned update_layer_model_version(const std::string skey);
    
    /**
     * Notify that using the given model version has finished its usage,
     * and is safe to update its model parameters.
     * 
     * @param skey persistent key.
     * @param iter_idx persistent key.
    */
    void notify_layer_usage_finished(const std::string skey, unsigned iter_idx);
    
    /** 
     * Apply `notify_layer_usage_finished` for all layer, using the model version saved at `layer_alloc_cnt_map_`
     * @param skey persistent key.
     * @returns current model version before update (i.e. iteration index)
    */
    void notify_all_layer_usage_finished();

    /** 
     * Increments the micro iteration of the given layer.
     * @param skey persistent key.
     * @returns micro iteration index before update (i.e. uiter index)
    */
    int update_micro_iteration(const std::string skey);


    /**
     * Retrieve CPU shared tensor entry.
     * Allocate new one if not exists. 
     * 
     * @param iter_idx iteration idx.
     * @param skey persistent key.
     * @param tensor_shape shape of the tensor.
    */
    SharedCpuTensorEntry *retrieve_cpu_shared_tensor_entry(int iter_idx, const std::string skey, c10::IntArrayRef tensor_shape);

    /**
     * Retrieve TrainTaskSharedProps from shared memory.
     * Allocate new one if not exists. 
     * 
     * @param key key.
     * @returns pointer to allocated TrainTaskSharedProps.
    */
    TrainTaskSharedProps *retrieve_train_task_shared_props(const std::string key);


    torch::Tensor & get_gpu_param_tensor(std::string skey);
    torch::Tensor & get_cpu_param_tensor(std::string skey);
};

#endif