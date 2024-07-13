#ifndef COMPRESS_COMPRESSOR_H
#define COMPRESS_COMPRESSOR_H

#include <string>
#include <cstdint>
#include <cstddef>
#include <utility>
#include <functional>
#include "../engine/threadpool.h"


class Compressor {

protected:
    std::unique_ptr<ThreadPool> &thread_pool_;
    std::string name_;

public:
    
    template<typename T = float>
    using Segment = std::pair<T *, size_t>;
    
    template<typename T = float>
    using ConstSegment = std::pair<const T *, const size_t>;

    Compressor(std::unique_ptr<ThreadPool> &thread_pool, std::string name) : thread_pool_(thread_pool), name_(name) {}

    inline const std::string &name() { return name_; }

    virtual size_t compress(const std::string &name, ConstSegment<float> src, uint32_t k, Segment<uint32_t> dst_idx, Segment<float> dst_val, int32_t idx_offset = 0) = 0;
};

#endif