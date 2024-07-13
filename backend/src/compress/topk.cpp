#include "topk.h"
#include <cmath>
#include <cstring>
#include <cassert>
#include <memory>
#include <algorithm>
#include <queue>
#include <vector>

// using Segment = Compressor::Segment;
// using ConstSegment = Compressor::ConstSegment;

size_t TopkCompressor::compress(const std::string &name, ConstSegment<float> src, uint32_t k, Segment<uint32_t> dst_idx, Segment<float> dst_val, int32_t idx_offset) {
    switch (method_) {
        case NTH_ELEMENT_MULTICORE:
            impl_nth_element_multicore(src, k, dst_idx, dst_val);
            break;
        case NTH_ELEMENT:
            impl_nth_element(src, k, dst_idx, dst_val);
            break;
        case HEAP:
            impl_heap(src, k, dst_idx, dst_val);
            break;
    }
    return dst_idx.second;
}

inline void TopkCompressor::impl_nth_element(const ConstSegment<float> &src, const uint32_t k, const Segment<uint32_t> &dst_idx, const Segment<float> &dst_val, int32_t idx_offset) {

    auto src_clone = std::make_unique<float []>(src.second);
    memcpy(src_clone.get(), src.first, src.second);
    
    if (dst_idx.second < k || dst_val.second < k)
        throw std::runtime_error("Invalid parameter k");

    std::nth_element(src_clone.get(), src_clone.get() + k, src_clone.get() + src.second, 
        [] (float d1, float d2) -> bool { return std::abs(d1) > std::abs(d2); 
    });
    

    for (uint32_t i = 0; i < k; i++) {
        dst_idx.first[i] = i + idx_offset;
        dst_val.first[i] = src_clone.get()[i];
    }

}

inline void TopkCompressor::impl_nth_element_multicore(const ConstSegment<float> &src, const uint32_t k, const Segment<uint32_t> &dst_idx, const Segment<float> &dst_val) {
    const size_t total_len = src.second;
    const size_t per_core_len = ((total_len + thread_pool_->n_threads() - 1) / thread_pool_->n_threads() + 15) / 16 * 16;
    const size_t per_core_k = (k + thread_pool_->n_threads() - 1) / thread_pool_->n_threads();

    
    std::vector<std::future<void>> futures;
    futures.reserve(total_len / per_core_len + 1);
    for (size_t acc_k = 0, idx = 0; acc_k < k && idx < total_len; acc_k += per_core_k, idx += per_core_len) {
        Segment<uint32_t> seg_idx = std::make_pair(dst_idx.first + acc_k, dst_idx.second + per_core_len);
        Segment<float> seg_val = std::make_pair(dst_val.first + acc_k, dst_val.second + per_core_len);
        impl_nth_element(src, std::min(per_core_k, k - acc_k), seg_idx, seg_val);

        futures.emplace_back(this->thread_pool_->enqueue([this, &seg_idx, &seg_val, &src, per_core_k, k, acc_k] () {
            this->impl_nth_element(src, std::min(per_core_k, k - acc_k), seg_idx, seg_val);
        }));
    }
    
    for (auto &fut : futures) 
        fut.wait();
}

inline void TopkCompressor::impl_heap(const ConstSegment<float> &src, const uint32_t k, const Segment<uint32_t> &dst_idx, const Segment<float> &dst_val) {

    struct Compare{
        bool operator () (const std::pair<float, uint32_t>& lhs, const std::pair<float, uint32_t>& rhs) const{
            return lhs.first > rhs.first;
        }
    };
    
    std::priority_queue<std::pair<float, uint32_t>, std::vector<std::pair<float, uint32_t>>, Compare> q;
    
    for(size_t i = 0; i < src.second; i++){
        q.push(std::make_pair(src.first[i], i));
        if(q.size() > k){
            q.pop();
        }
    }
    
    uint32_t i = 0;
    while(!q.empty()){
        auto e = q.top();
        dst_idx.first[i] = e.second;
        dst_val.first[i] = e.first;
        i += 1;
        q.pop();
    }
}