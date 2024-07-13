
#include "thresholdv.h"

#include <cassert>
#include <cstring>
#include <immintrin.h>
#include <iostream>
#include <algorithm>

#define PRINTING_DEBUG 0

static const uint32_t SIMD_BLOCK = sizeof(__m512) / sizeof(uint32_t); // 16

static inline bool is_aligned(const void *ptr) {
    return (uint64_t)ptr % SIMD_BLOCK == 0;
}

size_t ThresholdvCompressor::compress(const std::string &name, ConstSegment<float> src, uint32_t k, Segment<uint32_t> dst_idx, Segment<float> dst_val, int32_t idx_offset) {
    return impl_simd(src, k, dst_idx, dst_val, idx_offset);
}

/**
 * Uses nth-element to obtain top k threshold value.
 * 
 * This operation has high time complexity, i.e., worst case `O(n^2)`.
*/
float ThresholdvCompressor::impl_get_first_threshold (const ConstSegment<float> &src, const uint32_t k) {
    
    // make a copy of src
    auto src_copy = std::make_unique<float[]>(src.second);
    std::memcpy(src_copy.get(), src.first, sizeof(float) * src.second);


    std::nth_element(src_copy.get(), &src_copy[k], &src_copy[src.second], [] (float d1, float d2) -> bool { return std::abs(d1) > std::abs(d2); });

    return std::abs(src_copy[k]);
}


inline size_t ThresholdvCompressor::impl_naive (const ConstSegment<float> &src, const uint32_t k, const Segment<uint32_t> &dst_idx, const Segment<float> &dst_val, int32_t idx_offset) {
    assert(k > 0);
    
    // if threshold does not exist in the map, get the first threshold
    const uintptr_t ptr_key = reinterpret_cast<uintptr_t>(src.first);
    float threshold = 0;
    if (threshold_map_.find(ptr_key) == threshold_map_.end()) {
        threshold = impl_get_first_threshold(src, k);
        // std::cout << "Obtaining First Threshold for tensor " << ptr_key << "with threshold " << threshold << std::endl;
    } else {
        threshold = threshold_map_[ptr_key];
        // std::cout << "Reusing Threshold for tensor " << ptr_key << "with threshold " << threshold << std::endl;
    }

    size_t cnt_found = 0;
    float grad_abs_max = -1;
    for (size_t i = 0; i < src.second; ++i) {
        const auto grad_abs = std::abs(src.first[i]);
        grad_abs_max = std::max(grad_abs, grad_abs_max);
        if (grad_abs >= threshold) {
            if (cnt_found < dst_idx.second) {
                // add to dst_idx;
                dst_idx.first[cnt_found] = i;
                dst_val.first[cnt_found] = src.first[i];
            }
            cnt_found++;
        }
    }

    // std::cout << dst_idx.second << " is max number but found element is " << cnt_found << std::endl;

    // evaluate threshold is properly set, using AIMD
    if (k > cnt_found) {
        // must drop less
        threshold *= 0.99;
    } else if (k < cnt_found) {
        // must drop more
        threshold = threshold + 0.01 * cnt_found / k * grad_abs_max;
    }
    // update the threshold
    threshold_map_[ptr_key] = threshold;

    return std::min(cnt_found, dst_idx.second);
}

#if __AVX512F__ 
// https://stackoverflow.com/questions/36932240/avx2-what-is-the-most-efficient-way-to-pack-left-based-on-a-mask
static inline std::pair<size_t, float> filter_threshold(float *__restrict__ dst, uint32_t *__restrict__ dst_index, const float *__restrict__ src,  size_t len, float v, uint32_t idx_offset) {

    const float *endp = src+len;
    float *dst_start = dst;
    float max_abs_grad = -1;
    __m512 threshold = _mm512_set1_ps(v);
    __m512i indices = _mm512_set_epi32(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
    indices = _mm512_add_epi32(indices, _mm512_set1_epi32(idx_offset));
    __m512i increment = _mm512_set1_epi32(16);

    while (src + 16 <= endp) {
        __m512      sv  = _mm512_loadu_ps(src);
        __m512      sv2 = _mm512_sub_ps(sv, threshold);
        __mmask16 keep  = _mm512_cmp_ps_mask(sv2, _mm512_setzero_ps(), _CMP_GE_OQ);  // true for src >= 0.0, false for unordered and src < 0.0

        _mm512_mask_compressstoreu_ps(dst, keep, sv);
        _mm512_mask_compressstoreu_epi32(dst_index, keep, indices); 

        auto len = _mm_popcnt_u64(keep);

        indices = _mm512_add_epi32(indices, increment);
        src += 16;
        dst += len;
        dst_index += len;

    }
    return std::make_pair(dst - dst_start, max_abs_grad);
}

static inline float horizontal_max_avx512(__m512 vec) {

    // Reduce to 256-bit vector
    __m256 upper = _mm512_extractf32x8_ps(vec, 1);
    __m256 lower = _mm512_castps512_ps256(vec);
    __m256 max_vec = _mm256_max_ps(upper, lower);

    // Reduce to 128-bit vector
    __m128 high = _mm256_extractf128_ps(max_vec, 1);
    __m128 low = _mm256_castps256_ps128(max_vec);
    __m128 max_vec_128 = _mm_max_ps(high, low);

     // Reduce to scalar
    max_vec_128 = _mm_max_ps(max_vec_128, _mm_shuffle_ps(max_vec_128, max_vec_128, _MM_SHUFFLE(2, 3, 0, 1)));
    max_vec_128 = _mm_max_ps(max_vec_128, _mm_shuffle_ps(max_vec_128, max_vec_128, _MM_SHUFFLE(1, 0, 3, 2)));

    return _mm_cvtss_f32(max_vec_128);
}
#endif


inline size_t ThresholdvCompressor::impl_simd (const ConstSegment<float> &src, const uint32_t k, const Segment<uint32_t> &dst_idx, const Segment<float> &dst_val, int32_t idx_offset) {

#if __AVX512F__ 
    // if threshold does not exist in the map, get the first threshold
    const uintptr_t ptr_key = reinterpret_cast<uintptr_t>(src.first);
    float threshold = 0;
    if (threshold_map_.find(ptr_key) == threshold_map_.end()) {
        threshold = impl_get_first_threshold(src, k);
        // std::cout << "Obtaining First Threshold for tensor " << ptr_key << std::endl;
    } else {
        threshold = threshold_map_[ptr_key];
        // std::cout << "Reusing Threshold for tensor " << ptr_key << std::endl;
    }

    // var for primary gradient storing
    /*********************************************************************/
    const float *ptr_src = src.first;
    const float *const ptr_src_end = src.first + src.second;

    const uint32_t dst_len = dst_idx.second;
    uint32_t *ptr_dst_idx = dst_idx.first;
    float *ptr_dst_val = dst_val.first;

    const float traversal_ratio_threshold = 0.8;

    float max_abs_grad = -1;
    uint32_t cnt_found = 0;
    uint32_t cnt_found_original = 0;
    /*********************************************************************/
    
    // var for secondary gradient storing
    /*********************************************************************/
    auto secondary_gradient_idx = SIMD_BLOCK * (dst_idx.second / SIMD_BLOCK + 1);
    uint32_t *copy_dst_idx = new uint32_t[secondary_gradient_idx];
    float *copy_dst_val = new float[secondary_gradient_idx];

    memset(copy_dst_idx, 0, sizeof(uint32_t) * secondary_gradient_idx);
    memset(copy_dst_val, 0, sizeof(float) * secondary_gradient_idx);

    // initialization of secondary gradient
    auto init_block = dst_idx.second > SIMD_BLOCK ? SIMD_BLOCK : dst_idx.second;
    for (uint32_t i = 0; i < init_block; i++) copy_dst_idx[i] = i + idx_offset;
    memcpy(copy_dst_val, src.first, sizeof(float) * init_block);

    uint32_t *ptr_copy_dst_idx = copy_dst_idx;
    float *ptr_copy_dst_val = copy_dst_val;
    const uint32_t *ptr_copy_dst_idx_end = copy_dst_idx + secondary_gradient_idx;
    /*********************************************************************/
    
    assert((uint64_t)ptr_dst_val % SIMD_BLOCK == 0);
    assert((uint64_t)ptr_dst_idx % SIMD_BLOCK == 0);
    assert((uint64_t)ptr_src % SIMD_BLOCK == 0);
    
    // SIMD var setting
    /*********************************************************************/
    __m512 vec_threshold = _mm512_set1_ps(threshold);
    __m512i vec_indices = _mm512_set_epi32(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
            vec_indices = _mm512_add_epi32(vec_indices, _mm512_set1_epi32(idx_offset));
    __m512i vec_increment = _mm512_set1_epi32(16);
    /*********************************************************************/

    // Enable SIMD when src and dst has more than 16 elements to go
    while (ptr_src <= ptr_src_end - 16 && cnt_found + 16 <= dst_len) {
        __m512      sv  = _mm512_loadu_ps(ptr_src);
        __m512      sv1  = _mm512_abs_ps(sv);
        __m512      sv2 = _mm512_sub_ps(sv1, vec_threshold);
        __mmask16 keep  = _mm512_cmp_ps_mask(sv2, _mm512_setzero_ps(), _CMP_GE_OQ);  // true for src >= 0.0, false for unordered and src < 0.0
        __mmask16 keep_reversal  = ~keep;

        auto len = _mm_popcnt_u64(keep);
        auto len_reversal = SIMD_BLOCK - len;

        _mm512_mask_compressstoreu_ps(ptr_dst_val, keep, sv);
        _mm512_mask_compressstoreu_epi32(ptr_dst_idx, keep, vec_indices);

        if (ptr_copy_dst_idx + len_reversal <= ptr_copy_dst_idx_end) {
            _mm512_mask_compressstoreu_ps(ptr_copy_dst_val, keep_reversal, sv);
            _mm512_mask_compressstoreu_epi32(ptr_copy_dst_idx, keep_reversal, vec_indices); 
        }

        max_abs_grad = std::max(max_abs_grad, horizontal_max_avx512(sv));
        
        vec_indices = _mm512_add_epi32(vec_indices, vec_increment);
        ptr_src += 16;
        ptr_dst_idx += len;
        ptr_dst_val += len;
        ptr_copy_dst_idx += len_reversal;
        ptr_copy_dst_val += len_reversal;
        cnt_found += len;
    }

    auto traversal_ratio = (ptr_src - src.first) / (float)src.second;
    auto find_ratio = cnt_found / (float)dst_len;
    
#if PRINTING_DEBUG
    std::cout << "exit simd with traversal " << traversal_ratio * 100 << "% (found - " << find_ratio * 100 << "%)" << std::endl;
#endif

    for (size_t i = ptr_src - src.first; i < src.second && cnt_found < dst_len; i++) {
        const auto grad_abs = std::abs(src.first[i]);
        max_abs_grad = std::max(grad_abs, max_abs_grad);
        if (grad_abs >= threshold) {
            dst_idx.first[cnt_found] = i + idx_offset;
            dst_val.first[cnt_found++] = src.first[i];
        }
    }

#if PRINTING_DEBUG
    // std::cout << "exit first if with traversal " << traversal_ratio * 100 << "% (found - " << find_ratio * 100 << "%)" << std::endl;
#endif

    {
        int idx = 0;
        while (cnt_found < dst_len) {
            assert (copy_dst_val[idx] == src.first[copy_dst_idx[idx] - idx_offset]);
            dst_idx.first[cnt_found] = copy_dst_idx[idx];
            dst_val.first[cnt_found++] = copy_dst_val[idx++];
        }
        assert (cnt_found == dst_len);
        delete [] copy_dst_idx;
        delete [] copy_dst_val;
    }

#ifdef ELEM_DEBUG
    int idx = 0;
    while (idx < dst_len) {
        assert (dst_val.first[idx] == src.first[dst_idx.first[idx] - idx_offset]);
        idx++;
    }
#endif

#if PRINTING_DEBUG
    std::cout << dst_idx.second << " is max number but found element is " << cnt_found << " from thresholdv" <<std::endl;
#endif

    // evaluate threshold is properly set, using AIMD
    // if (traversal_ratio > traversal_ratio_threshold) {
    //     // must drop less
    //     threshold *= 0.99;
    // } else if (traversal_ratio < traversal_ratio_threshold) {
    //     // must drop more
    //     threshold = threshold + 0.01 * max_abs_grad;
    // }

    if (traversal_ratio > find_ratio) {
        // must drop less
        threshold *= 0.99;
    } else if (traversal_ratio < find_ratio) {
        // must drop more
        threshold = threshold + 0.01 * max_abs_grad;
    }
    // update the threshold
    threshold_map_[ptr_key] = threshold;

    return std::min(cnt_found, dst_len);
#else
    return impl_naive(src, k, dst_idx, dst_val, idx_offset);
#endif
}