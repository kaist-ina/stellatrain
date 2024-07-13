

#ifndef COMPRESS_RANDOMK_H
#define COMPRESS_RANDOMK_H

#include "compressor.h"


class RandomkCompressor : public Compressor {

private: 
    inline void 
    impl_simd (const ConstSegment<float> &src, const uint32_t k, const Segment<uint32_t> &dst_idx, const Segment<float> &dst_val, uint32_t idx_offset = 0);
    
public:
    bool multicore_;

    RandomkCompressor (std::unique_ptr<ThreadPool> &thread_pool, bool multicore = true)
     : multicore_(multicore), Compressor(thread_pool, "RandomK") {}
    virtual size_t compress(ConstSegment<float> src, uint32_t k, Segment<uint32_t> dst_idx, Segment<float> dst_val, int32_t idx_offset = 0);
};

#endif