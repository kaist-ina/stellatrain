#ifndef _ENGINE_SHM_MANAGER_H
#define _ENGINE_SHM_MANAGER_H

#include <torch/torch.h>
#include <torch/extension.h>
#include <fcntl.h>
#include <unistd.h>
#include <semaphore.h>
#include <unordered_map>
#include <mutex>
#include "config.h"


struct alignas(4) BarrierEntry {
    bool init;
    char name[MAX_SHM_BARRIER_NAME_LEN + 1];
    pthread_barrier_t barrier;
};


struct alignas(4) MemoryPoolMetaHeader {
    uint32_t num_shm_chunks;
    uint32_t num_entry[MAX_SHM_CHUNKS]; // can have up to 64 GB of shm
    pthread_barrier_t barrier;
    bool barrier_init;

    uint32_t num_shared_tensor_entry;

    size_t metadata_alloc_bytes;
    off64_t payload_offset;
};

struct alignas(4) MemoryPoolEntry {
    bool valid;
    uint32_t size;
    uint32_t next;
    uint32_t offset;
};


class ShmManager {

private:
    uint8_t *shm_ptr_ [MAX_SHM_CHUNKS];
    uint8_t *shm_meta_ptr_;
    sem_t* shm_meta_semaphore_;
    bool is_master_;
    pid_t local_session_id_;
    int local_rank_;

    void init_shared_memory_master();
    void init_shared_memory_slave();
    void shm_grow();
    void lock();
    void unlock();

public:
    ShmManager(bool is_master = true, pid_t master_pid = 0, int local_rank = 0) : 
        shm_ptr_{}, shm_meta_ptr_(nullptr), shm_meta_semaphore_(nullptr), is_master_(false), 
        local_session_id_(0), local_rank_(local_rank) {
        local_session_id_ = master_pid;
        if (is_master || master_pid == 0) {
            is_master_ = true;
            init_shared_memory_master();
        } else {
            is_master_ = false;
            init_shared_memory_slave();
        }
    }

    ~ShmManager();

    inline MemoryPoolMetaHeader * meta_header() const { return reinterpret_cast<MemoryPoolMetaHeader *>(shm_meta_ptr_); }
    
    torch::Tensor tensor_from(off64_t offset, at::IntArrayRef size);
    torch::Tensor alloc(at::IntArrayRef size, bool int32 = false, bool align = false);
    off64_t get_offset(void *ptr);
    off64_t get_meta_offset(void *ptr) const;    
    void * from_offset(off64_t offset) const;
    void * from_meta_offset(off64_t offset) const;
    void barrier(size_t size);
    void *malloc(size_t size, bool align = false);
    void *malloc_meta(size_t size);
    void free(void *ptr);
    void free_meta(void *ptr);

    
};
#endif