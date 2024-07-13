

#include <csignal>
#include <stdexcept>
#include <iostream>
#include "threadpool.h"

#if USE_LOWER_PRIORITY_FOR_WORKERS
#include <unistd.h>
#endif

thread_local int worker_id = -1;
int num_workers = 0;

int ThreadPool::worker_thread_main(size_t thread_idx) {
    sigset_t mask;
    sigemptyset (&mask);
    sigaddset (&mask, SIGTERM);
    sigaddset (&mask, SIGHUP);
    pthread_sigmask(SIG_BLOCK, &mask, NULL);

    char name[64] = {0, };
    snprintf(name, 63, "Worker-%lu", thread_idx);
    pthread_setname_np(pthread_self(), name);

#if USE_LOWER_PRIORITY_FOR_WORKERS
    nice(1);
#endif


    worker_id = thread_idx;

    while (true) {
        std::function<void()> task = [] () -> void {};
        {
            std::unique_lock<std::mutex> ul(m_);
            while (task_queue_.empty() && !finished_)
                cv_.wait(ul, [&] { return !task_queue_.empty() || finished_; });
            if (finished_)
                break;

            const auto& pair = task_queue_.top();
            task = std::move(pair.second); 
            task_queue_.pop();
        }
        task();
    }

    return 0;
}

ThreadPool::ThreadPool(unsigned int n_threads) : n_threads_(n_threads), finished_(false) {

    if (n_threads_ == 0)
        n_threads_ = std::thread::hardware_concurrency();

    if (n_threads_ <= 1) {
        throw std::runtime_error("This hardware does not support hardware concurrency.");
    }

    num_workers = n_threads_;
    
    /* Spawn threads */
    std::cerr << "Spawning " << n_threads_ << " worker threads...\n";

    arr_threads_.reserve(n_threads_);
    for (unsigned int i = 0; i < n_threads_; ++i) {
        arr_threads_.emplace_back(std::thread(&ThreadPool::worker_thread_main, this, i));
    }

}

ThreadPool::~ThreadPool() {
    finished_ = true;

    cv_.notify_all();
    for (auto & thread : arr_threads_) {
        thread.join();
    }
}
