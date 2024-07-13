

#ifndef COMPRESS_RANDOMK_H
#define COMPRESS_RANDOMK_H

#include "compressor.h"
#include <unordered_map>

class ThresholdvCompressor : public Compressor {

private: 
    std::unordered_map<uintptr_t, float> threshold_map_;

    inline size_t 
    impl_naive (const ConstSegment<float> &src, const uint32_t k, const Segment<uint32_t> &dst_idx, const Segment<float> &dst_val, int32_t idx_offset = 0);

    inline size_t 
    impl_simd (const ConstSegment<float> &src, const uint32_t k, const Segment<uint32_t> &dst_idx, const Segment<float> &dst_val, int32_t idx_offset = 0);

    float
    impl_get_first_threshold (const ConstSegment<float> &src, const uint32_t k);
    
public:
    bool multicore_;

    ThresholdvCompressor (std::unique_ptr<ThreadPool> &thread_pool, bool multicore = true)
     : multicore_(multicore), Compressor(thread_pool, "Thresholdv") {}
    virtual size_t compress(const std::string &name, ConstSegment<float> src, uint32_t k, Segment<uint32_t> dst_idx, Segment<float> dst_val, int32_t idx_offset = 0);
};

#endif