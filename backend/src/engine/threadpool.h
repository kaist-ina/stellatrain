#ifndef ENGINE_THREADPOOL_H
#define ENGINE_THREADPOOL_H

#include <vector>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <functional>
#include <future>
#include <iostream>
#include "config.h"


// https://modoocode.com/285


class ThreadPool {
    private:
        using PairType = std::pair<int, std::function<void()>>;
        struct Compare {
            bool operator()(const PairType& a, const PairType& b) {
                return a.first > b.first;
            }
        };


        unsigned int n_threads_;
        bool finished_;
        std::vector<std::thread> arr_threads_;
        std::priority_queue<PairType, std::vector<PairType>, Compare> task_queue_;
        std::mutex	m_;
        std::condition_variable		cv_;
        std::mutex	idle_m_;
        std::condition_variable		idle_cv_;

        int worker_thread_main(size_t thread_idx);

    public:
        ThreadPool(unsigned int n_threads = 0);
        ~ThreadPool();

        template <class F, class... Args> std::future<typename std::result_of<F(Args...)>::type> 
        enqueue(F&& f, Args&&... args);

        template <class F, class... Args> std::future<typename std::result_of<F(Args...)>::type> 
        enqueue_priority(int priority, F&& f, Args&&... args);

        inline unsigned int n_threads() const { return n_threads_; }
};


template <class F, class... Args>
std::future<typename std::result_of<F(Args...)>::type> ThreadPool::enqueue(F&& f, Args&&... args) {
    
    if (finished_) {
        throw std::runtime_error("ThreadPool has been terminated.");
    }
    std::cerr << "Enqueue using no priority is not recommended" << std::endl;

    using return_type = typename std::result_of<F(Args...)>::type;

    auto task = std::make_shared<std::packaged_task<return_type()>>(std::bind(std::forward<F>(f), std::forward<Args>(args)...));
    
    std::future<return_type> job_result_future = task->get_future();
    {
        std::lock_guard<std::mutex> lock(m_);
        task_queue_.emplace(0, [task]() { (*task)(); });
    }
    cv_.notify_one();

    return job_result_future;
}


template <class F, class... Args>
std::future<typename std::result_of<F(Args...)>::type> ThreadPool::enqueue_priority(int priority, F&& f, Args&&... args) {
    
    if (finished_) {
        throw std::runtime_error("ThreadPool has been terminated.");
    }

    using return_type = typename std::result_of<F(Args...)>::type;

    auto task = std::make_shared<std::packaged_task<return_type()>>(std::bind(std::forward<F>(f), std::forward<Args>(args)...));
    
    std::future<return_type> job_result_future = task->get_future();
    {
        std::lock_guard<std::mutex> lock(m_);
#if PRIORITY_SCHED
        task_queue_.emplace(priority, [task]() { (*task)(); });
#else
        task_queue_.emplace(0, [task]() { (*task)(); });
#endif
    }
    cv_.notify_one();

    return job_result_future;
}

#endif