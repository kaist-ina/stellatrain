

#ifndef COMPRESS_THRESHOLDV16_H
#define COMPRESS_THRESHOLDV16_H

#include "compressor.h"
#include <unordered_map>

class ThresholdvCompressor16 : public Compressor {

private: 
    std::mutex threshold_map_mutex_;
    std::unordered_map<std::string, float> threshold_map_;
    std::unordered_map<std::string, float> threshold_map_inc_;

    inline size_t 
    impl_naive (const ConstSegment<float> &src, const uint32_t k, const Segment<uint32_t> &dst_idx, const Segment<float> &dst_val, int32_t idx_offset = 0);

    inline size_t 
    impl_simd_test_thrshold (const ConstSegment<float> &src, const uint32_t k, const Segment<uint32_t> &dst_idx, const Segment<float> &dst_val, uint32_t idx_offset = 0);

    inline size_t 
    impl_simd (const ConstSegment<float> &src, const uint32_t k, const Segment<uint32_t> &dst_idx, const Segment<float> &dst_val, int32_t idx_offset = 0);

    inline size_t 
    impl_simd_v2 (const std::string &name, const ConstSegment<float> &src, const uint32_t k, const Segment<uint32_t> &dst_idx, const Segment<float> &dst_val, int32_t idx_offset = 0);


    float
    impl_get_first_threshold (const ConstSegment<float> &src, const uint32_t k);
    
public:
    bool multicore_;

    ThresholdvCompressor16 (std::unique_ptr<ThreadPool> &thread_pool, bool multicore = true)
     : multicore_(multicore), Compressor(thread_pool, "Thresholdv16") {}
    virtual size_t compress(const std::string &name, ConstSegment<float> src, uint32_t k, Segment<uint32_t> dst_idx, Segment<float> dst_val, int32_t idx_offset = 0);
};

#endif