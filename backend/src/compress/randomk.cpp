#include "randomk.h"
#include "../misc/simdxorshift128plus.h"

#include <cmath>
#include <cstring>
#include <cassert>
#include <memory>
#include <algorithm>
#include <queue>
#include <vector>
#include <immintrin.h>
#include <iostream>
// using Compressor::Segment;


// template<typename T>
// using Segment<T> = Compressor::Segment<T>;

// template<typename T>
// using ConstSegment<T> = Compressor::ConstSegment<T>;

static const uint32_t SIMD_BLOCK = sizeof(__m256i) / sizeof(uint32_t); // 8

static inline bool is_aligned(const void *ptr) {
    return (uint64_t)ptr % SIMD_BLOCK == 0;
}

size_t RandomkCompressor::compress(ConstSegment<float> src, uint32_t k, Segment<uint32_t> dst_idx, Segment<float> dst_val, int32_t idx_offset) {
    
    assert(idx_offset == 0); // not implemented

    if (multicore_) {
        const size_t total_len = src.second;
        const auto num_threads = std::min(thread_pool_->n_threads(), k);
        const size_t per_core_len = ((total_len + thread_pool_->n_threads() - 1) / thread_pool_->n_threads() + 15) / 16 * 16;
        const size_t per_core_k = (k + thread_pool_->n_threads() - 1) / thread_pool_->n_threads() / 8 * 8;

        
        std::vector<std::future<void>> futures;
        futures.reserve(total_len / per_core_len + 1);
        size_t from_i = 0, from_j = 0;

        for (size_t thread_idx = 0; thread_idx < num_threads; thread_idx++) {
            
            size_t target_i = ((thread_idx + 1) * total_len) / num_threads;
            size_t target_j = ((thread_idx + 1) * k + num_threads - 1) / num_threads;
            
            target_i = std::max(from_i + target_j - from_j, target_i);
            assert(target_i - from_i >= target_j - from_j);
            
            if (thread_idx + 1 < num_threads) {
               while (!is_aligned(src.first + target_i) && target_i < total_len)
                    target_i += 1;
                while (!is_aligned(dst_idx.first + target_j) && target_j < total_len)
                    target_j += 1;
            }

            target_i = std::min(total_len, target_i);
            target_j = std::min((size_t)k, target_j);
            
            const auto local_ptr = src.first + from_i;
            const auto local_len = std::min(src.second - from_i, target_i - from_i);
            const auto local_k = std::min(k - from_j, target_j - from_j);
            const auto local_ptr_indices = dst_idx.first + from_j;
            const auto local_ptr_values = dst_val.first + from_j;

            if (local_len == 0 && local_k == 0)
                break;

            if (local_k == 0)
                continue;

            assert(local_len != 0);
            assert(local_k != 0);
            assert(local_len >= local_k);
            
            assert((uint64_t)(local_ptr) % SIMD_BLOCK == 0);
            assert((uint64_t)(local_ptr_indices) % SIMD_BLOCK == 0);
            assert((uint64_t)(local_ptr_values) % SIMD_BLOCK == 0);

            ConstSegment<float> seg_src = std::make_pair(local_ptr, local_len);
            Segment<uint32_t> seg_idx = std::make_pair(dst_idx.first + from_j, local_k);
            Segment<float> seg_val = std::make_pair(dst_val.first + from_j, local_k);

            futures.emplace_back(this->thread_pool_->enqueue([this, seg_src, seg_idx, seg_val, from_i, local_k] () {
                this->impl_simd(seg_src, local_k, seg_idx, seg_val, from_i);
            }));

            from_i = target_i;
            from_j = target_j;
        }
        
        for (auto &fut : futures) 
            fut.wait();
    } else {
        impl_simd(src, k, dst_idx, dst_val);
    }

    return dst_idx.second;
}


// https://stackoverflow.com/questions/70558346/generate-random-numbers-in-a-given-range-with-avx2-faster-than-svml-mm256-rem
static inline __m256i narrowRandom( __m256i bits, int range )
{
    assert( range > 1 );

    // Convert random bits into FP32 number in [ 1 .. 2 ) interval
    const __m256i mantissaMask = _mm256_set1_epi32( 0x7FFFFF );
    const __m256i mantissa = _mm256_and_si256( bits, mantissaMask );
    const __m256 one = _mm256_set1_ps( 1 );
    __m256 val = _mm256_or_ps( _mm256_castsi256_ps( mantissa ), one );

    // Scale the number from [ 1 .. 2 ) into [ 0 .. range ),
    // the formula is ( val * range ) - range
    const __m256 rf = _mm256_set1_ps( (float)range );
    val = _mm256_fmsub_ps( val, rf, rf );

    // Convert to integers
    // The instruction below always truncates towards 0 regardless on MXCSR register.
    // If you want ranges like [ -10 .. +10 ], use _mm256_add_epi32 afterwards
    return _mm256_cvttps_epi32( val );
}

inline void RandomkCompressor::impl_simd (const ConstSegment<float> &src, const uint32_t k, const Segment<uint32_t> &dst_idx, const Segment<float> &dst_val, uint32_t idx_offset) {
    uint32_t i = 0;
    avx_xorshift128plus_key_t mykey;
    avx_xorshift128plus_init(324, 4444, &mykey);

    if (reinterpret_cast<uintptr_t>(dst_idx.first) % SIMD_BLOCK != 0) {
        throw std::runtime_error(std::string("Destination idx is not aligned : ") + std::to_string(reinterpret_cast<uintptr_t>(dst_idx.first)));
    }
    
    if (reinterpret_cast<uintptr_t>(dst_val.first) % SIMD_BLOCK != 0) {
        throw std::runtime_error("Destination value is not aligned");
    }
    
    if (reinterpret_cast<uintptr_t>(src.first) % SIMD_BLOCK != 0) {
        throw std::runtime_error("Source pointer is not aligned");
    }

    __m256i idx_offset_block = _mm256_set1_epi32(idx_offset);

    auto ptr = src.first;
    auto len = src.second;
    auto ptr_values = dst_val.first;
    auto ptr_indices = dst_idx.first;

    while (i + SIMD_BLOCK <= k) {
        __m256i rand_output = narrowRandom(avx_xorshift128plus(&mykey), len);
        _mm256_storeu_si256((__m256i *)(ptr_values + i), _mm256_i32gather_epi32((int *)ptr, rand_output, 4));
        _mm256_storeu_si256((__m256i *)(ptr_indices + i), _mm256_add_epi32(rand_output, idx_offset_block));
        i += SIMD_BLOCK;
    }

    if (i != k) {
        uint32_t buffer_idx[sizeof(__m256i) / sizeof(uint32_t)], buffer_val[sizeof(__m256i) / sizeof(uint32_t)];
        __m256i rand_output = narrowRandom(avx_xorshift128plus(&mykey), len);
        _mm256_storeu_si256((__m256i *)buffer_val, _mm256_i32gather_epi32((int *)ptr, rand_output, 4));
        _mm256_storeu_si256((__m256i *)buffer_idx, _mm256_add_epi32(rand_output, idx_offset_block));
        memcpy(ptr_indices + i, buffer_idx, sizeof(uint32_t) * (k - i));
        memcpy(ptr_values + i, buffer_val, sizeof(uint32_t) * (k - i));
    }
}