
#include <sys/mman.h>
#include <sys/stat.h>
#include <string>
#include <cstring>
#include <unistd.h>

#include <cuda_runtime_api.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>
#include "shm_manager.h"



static_assert(META_BYTES_ALLOC >= sizeof(MemoryPoolMetaHeader), "Allocated Metadata size is too small.");

void ShmManager::init_shared_memory_master() {

    if (local_session_id_ == 0)
        local_session_id_ = getpid();

    // allocate shared memory for metadata
    {
        // allocate semaphore
        auto sem_name = (SEM_NAME_PREFIX + std::to_string(local_session_id_));

        // try remove
        sem_unlink(sem_name.c_str());
        shm_meta_semaphore_ = sem_open(sem_name.c_str(), O_CREAT, 0666, 1);
        assert(shm_meta_semaphore_ != SEM_FAILED);

        lock();

        std::string shm_meta_name = SHM_NAME_PREFIX + "meta-" + std::to_string(local_session_id_);
        
        std::cout << "Starting SHM Master... at " << shm_meta_name << std::endl;

        // try remove
        shm_unlink(shm_meta_name.c_str());

        int fd = shm_open(shm_meta_name.c_str(), O_RDWR | O_CREAT, S_IRUSR | S_IWUSR);
        if (fd == -1) {
            perror("open");
            throw std::runtime_error("Cannot create shared memory segment");
        }

        if (ftruncate64(fd, META_BYTES_ALLOC) == -1) {
            perror("ftruncate");
            throw std::runtime_error("ftruncate failed");
        }
        
        uint8_t *addr = reinterpret_cast<uint8_t *>(mmap(NULL, META_BYTES_ALLOC, PROT_WRITE | PROT_READ, MAP_SHARED, fd, 0));
        if (addr == MAP_FAILED) {
            perror("mmap");
            throw std::runtime_error("mmap failed");
        }

        shm_meta_ptr_ = addr;
        memset(shm_meta_ptr_, 0, META_BYTES_ALLOC);
        auto *header = reinterpret_cast<MemoryPoolMetaHeader *>(shm_meta_ptr_);
        header->num_shm_chunks = 1;


    }

    // allocate shared memory for data
    {
        const std::string shm_name = SHM_NAME_PREFIX + "data-" + std::to_string(local_session_id_) + "-0";

        // try remove
        shm_unlink(shm_name.c_str());
        std::cout << "Starting SHM Master... at " << shm_name << std::endl;

        int fd = shm_open(shm_name.c_str(), O_RDWR | O_CREAT, S_IRUSR | S_IWUSR);
        if (fd == -1) {
            perror("open");
            throw std::runtime_error("Cannot create shared memory segment");
        }

        if (ftruncate64(fd, DATA_BYTES_ALLOC) == -1) {
            perror("ftruncate");
            throw std::runtime_error("ftruncate failed");
        }
        
        uint8_t *addr = reinterpret_cast<uint8_t *>(mmap(NULL, DATA_BYTES_ALLOC, PROT_WRITE | PROT_READ, MAP_SHARED, fd, 0));
        if (addr == MAP_FAILED) {
            perror("mmap");
            throw std::runtime_error("mmap failed");
        }

        shm_ptr_[0] = addr;
        cudaHostRegister(addr, DATA_BYTES_ALLOC, cudaHostRegisterMapped);
    }
    
    unlock();

    std::cout << "Starting SHM Master OK" << std::endl;
    
}

void ShmManager::lock() {
    if (sem_wait(shm_meta_semaphore_) != 0) {
        perror("sem_wait");
    }
}

void ShmManager::unlock() {
    if (sem_post(shm_meta_semaphore_) != 0) {
        perror("sem_wait");
    }
}

void ShmManager::init_shared_memory_slave() {

    {
        std::string shm_meta_name = SHM_NAME_PREFIX + "meta-" + std::to_string(local_session_id_);
        auto sem_name = (SEM_NAME_PREFIX + std::to_string(local_session_id_));        
        std::cout << "Starting SHM Slave... at " << shm_meta_name << std::endl;

        while (true) {
            shm_meta_semaphore_ = sem_open(sem_name.c_str(), O_CREAT, 0666, 1);
            if (shm_meta_semaphore_ == SEM_FAILED) {
                usleep(1000*100);
            } else {
                break;
            }
        }

        lock();
        unlock();

        int fd;
        while (true) {
            fd = shm_open(shm_meta_name.c_str(), O_RDWR, S_IRUSR | S_IWUSR);
            if (fd == -1) {
                usleep(1000*100);
            } else {
                break;
            }
        }

        // map shared memory to process address space
        uint8_t *addr = reinterpret_cast<uint8_t *>(mmap(NULL, DATA_BYTES_ALLOC, PROT_WRITE | PROT_READ, MAP_SHARED, fd, 0));
        if (addr == MAP_FAILED)
        {
            perror("mmap");
            throw std::runtime_error("mmap failed");
        }
        
        shm_meta_ptr_ = addr;
        
    }

    auto *header = reinterpret_cast<MemoryPoolMetaHeader *>(shm_meta_ptr_);

    unsigned mmaped_shm_cnt = 0;
    for (uint32_t chunk_idx = 0; chunk_idx < header->num_shm_chunks; chunk_idx++) {
        std::string shm_name = SHM_NAME_PREFIX + "data-" + std::to_string(local_session_id_) + "-" + std::to_string(chunk_idx);
        int fd;
        
        std::cout << "Starting SHM Slave... at " << shm_name << std::endl;

        while (true) {
            fd = shm_open(shm_name.c_str(), O_RDWR, S_IRUSR | S_IWUSR);
            if (fd == -1) {
                usleep(1000*100);
            } else {
                break;
            }
        }

        // map shared memory to process address space
        uint8_t *addr = reinterpret_cast<uint8_t *>(mmap(NULL, DATA_BYTES_ALLOC, PROT_WRITE | PROT_READ, MAP_SHARED, fd, 0));
        if (addr == MAP_FAILED)
        {
            perror("mmap");
            throw std::runtime_error("mmap failed");
        }
        
        shm_ptr_[chunk_idx] = addr;
        cudaHostRegister(addr, DATA_BYTES_ALLOC, cudaHostRegisterMapped);
        mmaped_shm_cnt += 1;
    }

    std::cout << "Starting SHM Slave OK, mmaped " << mmaped_shm_cnt << " data shms " << std::endl;
    assert(mmaped_shm_cnt > 0);
}


void ShmManager::barrier(size_t size) {
    auto *header = reinterpret_cast<MemoryPoolMetaHeader *>(shm_meta_ptr_);

    sem_wait(shm_meta_semaphore_);
    // now metadata is locked!
    bool master = false;

    if (!header->barrier_init) {
        pthread_barrierattr_t barrier_attr;
        pthread_barrierattr_setpshared(&barrier_attr, PTHREAD_PROCESS_SHARED);
        auto ret = pthread_barrier_init(&header->barrier, &barrier_attr, size);
        header->barrier_init = true;
        master = true;
    }

    sem_post(shm_meta_semaphore_);    

    pthread_barrier_wait(&header->barrier);
    std::cout << "Barrier OK" << std::endl;

    sem_wait(shm_meta_semaphore_);
    if (master) {
        header->barrier_init = false;
    }
    sem_post(shm_meta_semaphore_);    
}

void ShmManager::shm_grow() {
    
}



ShmManager::~ShmManager() {

    // free shared memory for data

    for (unsigned chunk_idx = 0; chunk_idx < MAX_SHM_CHUNKS; chunk_idx++) {
        if (shm_ptr_[chunk_idx] == nullptr)
            continue;

        C10_CUDA_CHECK(cudaHostUnregister(shm_ptr_[chunk_idx]));

        if (munmap(shm_ptr_[chunk_idx], DATA_BYTES_ALLOC) == -1) {
            perror("munmap");
        }

        if (is_master_) {
            const auto shm_name = SHM_NAME_PREFIX + "data-" + std::to_string(local_session_id_) + "-" + std::to_string(chunk_idx);
            if (shm_unlink(shm_name.c_str()) == -1) {
                std::cerr << shm_name << std::endl;
                perror("unlink");
            }
        }
        
    }

    sem_close(shm_meta_semaphore_);

    // free shared memory for meta
    if (shm_meta_ptr_ == nullptr)
        return;

    if (munmap(shm_meta_ptr_, META_BYTES_ALLOC) == -1) {
        perror("munmap");
    }

    if (is_master_) {
        const auto shm_name = SHM_NAME_PREFIX + "meta-" + std::to_string(local_session_id_);
        if (shm_unlink(shm_name.c_str()) == -1) {
            perror("unlink");
        }
    
        const auto sem_name = SEM_NAME_PREFIX + std::to_string(local_session_id_);
        sem_unlink(sem_name.c_str());
    }

}


torch::Tensor ShmManager::alloc(at::IntArrayRef size, bool int32, bool align) {

    size_t bytes = 4;
    for (const auto t : size) {
        bytes *= t;
    }

    void *ptr = malloc(bytes, align);
    const auto tensor = torch::from_blob(ptr, size, 
        at::TensorOptions().pinned_memory(true).dtype(int32 ? torch::kInt32 : torch::kFloat32)
    );

    assert(tensor.numel() > 0);
    assert(tensor.numel() * tensor.element_size() == bytes);

    return tensor;
}

torch::Tensor ShmManager::tensor_from(off64_t offset, at::IntArrayRef size) {

    const auto page = offset / DATA_BYTES_ALLOC;
    const auto page_offset = offset % DATA_BYTES_ALLOC;

    auto *header = reinterpret_cast<MemoryPoolMetaHeader *>(shm_meta_ptr_);
    void * ptr = shm_ptr_[page] + page_offset;
    return torch::from_blob(ptr, size, 
        at::TensorOptions().pinned_memory(true)
    );

}

off64_t ShmManager::get_offset(void *ptr) {
    auto *header = reinterpret_cast<MemoryPoolMetaHeader *>(shm_meta_ptr_);
    for (unsigned chunk_idx = 0; chunk_idx < header->num_shm_chunks; chunk_idx++) {
        if (ptr >= shm_ptr_[chunk_idx] && shm_ptr_[chunk_idx] + DATA_BYTES_ALLOC > ptr)
            return (reinterpret_cast<uint8_t *>(ptr) - shm_ptr_[chunk_idx]) + chunk_idx * DATA_BYTES_ALLOC;
    }

    return 0;
}

off64_t ShmManager::get_meta_offset(void *ptr) const {
    return reinterpret_cast<uint8_t *>(ptr) - reinterpret_cast<uint8_t *>(shm_meta_ptr_);
}

void * ShmManager::from_offset(off64_t offset) const {
    auto *header = reinterpret_cast<MemoryPoolMetaHeader *>(shm_meta_ptr_);
    const auto chunk_idx = offset / DATA_BYTES_ALLOC;
    const auto chunk_off = offset % DATA_BYTES_ALLOC;

    assert (chunk_idx < header->num_shm_chunks);
    
    return shm_ptr_[chunk_idx] + chunk_off;
}

void * ShmManager::from_meta_offset(off64_t offset) const {
    return reinterpret_cast<void *>(reinterpret_cast<uint8_t *>(shm_meta_ptr_) + offset);
}


void *ShmManager::malloc(size_t size, bool align) {

    // for now, really simple design even without entry
    
    auto *header = reinterpret_cast<MemoryPoolMetaHeader *>(shm_meta_ptr_);
    
    lock();
    void *ptr = nullptr;
    size_t total_allocated = 0;
    for (unsigned chunk_idx = 0; chunk_idx < header->num_shm_chunks; chunk_idx++) {
        size_t align_req = align ? ((16 - (header->num_entry[chunk_idx] % 16)) % 16) : 0; 
        total_allocated += header->num_entry[chunk_idx];

        if (header->num_entry[chunk_idx] + size + align_req > DATA_BYTES_ALLOC)
            continue;
        ptr = shm_ptr_[chunk_idx] + header->num_entry[chunk_idx] + align_req;
        // std::stringstream sstream;
        // sstream << "[" << getpid() << "] Allocating " << size << " Bytes (" << chunk_idx << ", " << header->num_entry[chunk_idx] << ")";
        // std::cerr << sstream.str() << std::endl;
        header->num_entry[chunk_idx] += size + align_req;
        break;
    }

    unlock();

    if (ptr == nullptr) {
        std::stringstream s;
        s << "Cannot allocate memory segment of size " << size << "! Already allocated " <<  total_allocated << " bytes!\n";
        s << "Chunk statistics:\n";

        for (unsigned chunk_idx = 0; chunk_idx < header->num_shm_chunks; chunk_idx++) {
            s << "Chunk #" << chunk_idx << ": " << header->num_entry[chunk_idx] << " / " << DATA_BYTES_ALLOC << "\n";
        }
        unlock();
        std::cerr << s.str() << std::endl;
        throw std::runtime_error(s.str());
    }


    return ptr;
}


void *ShmManager::malloc_meta(size_t size) {
    auto *header = reinterpret_cast<MemoryPoolMetaHeader *>(shm_meta_ptr_);
    lock();
    assert(header->metadata_alloc_bytes + sizeof(MemoryPoolMetaHeader) + size <= META_BYTES_ALLOC);
    auto *ptr = reinterpret_cast<uint8_t *>(header + 1) + header->metadata_alloc_bytes;
    header->metadata_alloc_bytes += size;
    unlock();
    return ptr;
}

void ShmManager::free(void *ptr) {
    std::cout << "Freeing ptr " << ptr << " \n";
    // basically do nothing!
    (void) ptr;
}

void ShmManager::free_meta(void *ptr) {
    std::cout << "Freeing meta ptr " << ptr << " \n";
    // basically do nothing!
    (void) ptr;
}