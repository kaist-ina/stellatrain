

#ifndef COMPRESS_TOPK_H
#define COMPRESS_TOPK_H

#include "compressor.h"


class TopkCompressor : public Compressor {

private: 

    inline void 
    impl_nth_element (const ConstSegment<float> &src, const uint32_t k, const Segment<uint32_t> &dst_idx, const Segment<float> &dst_val, int32_t idx_offset = 0);
    
    inline void 
    impl_nth_element_multicore (const ConstSegment<float> &src, const uint32_t k, const Segment<uint32_t> &dst_idx, const Segment<float> &dst_val);
    
    inline void 
    impl_heap (const ConstSegment<float> &src, const uint32_t k, const Segment<uint32_t> &dst_idx, const Segment<float> &dst_val);
    
public:
    enum TopkCompressMethod {
        NTH_ELEMENT_MULTICORE, NTH_ELEMENT, HEAP
    };
    
    TopkCompressMethod method_;

    TopkCompressor (std::unique_ptr<ThreadPool> &thread_pool)
     : method_(NTH_ELEMENT), Compressor(thread_pool, "Topk") {}
    virtual size_t compress(const std::string &name, ConstSegment<float> src, uint32_t k, Segment<uint32_t> dst_idx, Segment<float> dst_val, int32_t idx_offset = 0);
};

#endif