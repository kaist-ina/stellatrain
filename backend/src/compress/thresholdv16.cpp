
#include "thresholdv16.h"

#include <cassert>
#include <cstring>
#include <immintrin.h>
#include <iostream>
#include <algorithm>
#include <numeric>
#include <list>
#include <cmath>
#include <string>
#include <sstream> 

#define DATA_INPUT_SIZE 16
#define PRINTING_DEBUG 0
#define USE_NEW_IMPL 1
#define USE_AVX_512F 0

static const uint32_t CACHE_LINE_NUMEL = 16;

size_t ThresholdvCompressor16::compress(const std::string &name, ConstSegment<float> src, uint32_t k, Segment<uint32_t> dst_idx, Segment<float> dst_val, int32_t idx_offset) {

#if USE_NEW_IMPL
    return impl_simd_v2(name, src, k, dst_idx, dst_val, idx_offset);
#else
    return impl_simd(src, k, dst_idx, dst_val, idx_offset);
#endif
}

/**
 * Uses nth-element to obtain top k threshold value.
 * 
 * This operation has high time complexity, i.e., worst case `O(n^2)`.
*/
float ThresholdvCompressor16::impl_get_first_threshold (const ConstSegment<float> &src, const uint32_t k) {
    
    assert (src.second >= k);

    const auto block_len = (src.second + CACHE_LINE_NUMEL - 1) / CACHE_LINE_NUMEL;
    const auto block_k = k / CACHE_LINE_NUMEL;
    auto src_copy = std::make_unique<float []>(block_len);

    memset(src_copy.get(), 0, sizeof(float) * block_len);
    for (int i = 0; i < src.second; i++) {
        src_copy[i / CACHE_LINE_NUMEL] += std::abs(src.first[i]);
    }

    if (src.second % CACHE_LINE_NUMEL)
        src_copy[block_len - 1] *= static_cast<float>(CACHE_LINE_NUMEL) / (src.second % CACHE_LINE_NUMEL);

    std::nth_element(src_copy.get(), src_copy.get() + block_k, src_copy.get() + block_len, [] (float d1, float d2) -> bool { return std::abs(d1) > std::abs(d2); });
    return std::abs(src_copy[block_k]);
}


static inline float hsum_float_avx(__m256 v) {
    // __m128 vlow  = _mm256_castps256_ps128(v);
    // __m128 vhigh = _mm256_extractf128_ps(v, 1); // high 128
    // vlow  = _mm_add_ps(vlow, vhigh);     // reduce down to 128

    // __m128 high64 = _mm_unpackhi_ps(vlow, vlow);
    // return _mm_cvtss_f32(_mm_add_ps(vlow, high64));  // reduce to scalar

    const __m128 vlow  = _mm256_castps256_ps128(v);
    const __m128 vhigh = _mm256_extractf128_ps(v, 1); // high 128
    const __m128 sum128 = _mm_add_ps(vlow, vhigh);
    __m128 shuf = _mm_movehdup_ps(sum128); // broadcast elements 3,1 to 2,0
    __m128 sums = _mm_add_ps(sum128, shuf);
    shuf = _mm_movehl_ps(shuf, sums); // high half -> low half
    sums = _mm_add_ss(sums, shuf);
    return _mm_cvtss_f32(sums);
}

/**
 * AVX implementation of Thresholdv16. Use AVX implementation for now, instead of AVX512F which requires recent CPU.
*/
inline size_t ThresholdvCompressor16::impl_simd_v2 (const std::string &name, const ConstSegment<float> &src, const uint32_t k, const Segment<uint32_t> &dst_idx, const Segment<float> &dst_val, int32_t idx_offset) {
    
    // if threshold does not exist in the map, get the first threshold
    const std::string ptr_key = name;
    float threshold = 0, threshold_inc = 0;
    {
        std::unique_lock<std::mutex> ul(threshold_map_mutex_);
        
        if (threshold_map_.find(ptr_key) == threshold_map_.end()) {
            threshold = -1; 
        } else {
            threshold = threshold_map_[ptr_key];
            threshold_inc = threshold_map_inc_[ptr_key];
        }
    }

    if (threshold == -1) {
        threshold = impl_get_first_threshold(src, k);
        threshold_inc = threshold * 0.01;
    }

    const float *ptr_src = src.first;
    const float *ptr_src_end = src.first + src.second;
    const uint32_t dst_len = dst_idx.second;
    uint32_t *ptr_dst_idx = dst_idx.first;
    float *ptr_dst_val = dst_val.first;
    float max_abs_grad = -1;
    uint32_t cnt_found = 0;
    
    std::vector<std::pair<float, uint32_t>> vec_secondary_candidates;
    vec_secondary_candidates.reserve(src.second / 16 + 1);

    /**
     * Algorithm
     * 1. Filter out using threshold-v
     * 2. Is destination buffer full? return
     * 3. Filter out using min heap to fill remainder
    */

    /* Use SIMD */

#if USE_AVX_512F
    throw std::runtime_error("AVX_512F Not implemented");

#else
    static const uint32_t SIMD_BLOCK = sizeof(__m256) / sizeof(uint32_t); // 8

    __m256i vec_indices0 = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i vec_indices1 = _mm256_set_epi32(15, 14, 13, 12, 11, 10, 9, 8);
    vec_indices0 = _mm256_add_epi32(vec_indices0, _mm256_set1_epi32(idx_offset));
    vec_indices1 = _mm256_add_epi32(vec_indices1, _mm256_set1_epi32(idx_offset));
    __m256i vec_increment = _mm256_set1_epi32(CACHE_LINE_NUMEL);
    
    const float * ptr_src_end_simd = ptr_src + (src.second - (src.second % CACHE_LINE_NUMEL));
    uint32_t dst_end_simd = dst_len - (dst_len % CACHE_LINE_NUMEL);
    
    /* https://stackoverflow.com/questions/23847377/how-does-this-function-compute-the-absolute-value-of-a-float-through-a-not-and-a */
    const __m256 mask_abs = _mm256_set1_ps(-0.0);

    /* try SIMD first */
    while (ptr_src < ptr_src_end_simd && cnt_found < dst_end_simd) {
        __m256 vec_val0 = _mm256_loadu_ps(ptr_src);
        __m256 vec_val1 = _mm256_loadu_ps(ptr_src + SIMD_BLOCK);
        __m256 vec_val0_abs = _mm256_andnot_ps(mask_abs, vec_val0);
        __m256 vec_val1_abs = _mm256_andnot_ps(mask_abs, vec_val1);
        const float sum = hsum_float_avx(vec_val0_abs) + hsum_float_avx(vec_val1_abs);

        if (sum >= threshold) {
            _mm256_storeu_ps(ptr_dst_val, vec_val0);
            _mm256_storeu_ps(ptr_dst_val + SIMD_BLOCK, vec_val1);
            _mm256_storeu_si256(reinterpret_cast<__m256i_u *>(ptr_dst_idx), vec_indices0);
            _mm256_storeu_si256(reinterpret_cast<__m256i_u *>(ptr_dst_idx + SIMD_BLOCK), vec_indices1);
            ptr_dst_idx += CACHE_LINE_NUMEL;
            ptr_dst_val += CACHE_LINE_NUMEL;
            cnt_found += CACHE_LINE_NUMEL;
        } else {
            vec_secondary_candidates.push_back(std::make_pair(sum, ptr_src - src.first));
        }

        vec_indices0 = _mm256_add_epi32(vec_indices0, vec_increment);
        vec_indices1 = _mm256_add_epi32(vec_indices1, vec_increment);
        ptr_src += CACHE_LINE_NUMEL;
    }
    // std::cerr << "Stage 1: After SIMD, SRC = " << (ptr_src - src.first) << "/" << src.second << ", DST = " << cnt_found << "/" << dst_len << "\n";

    assert(ptr_src <= ptr_src_end);
    assert(cnt_found <= dst_len);

    /* Now either remaining src < 16 or remaining dst < 16 */

    if (ptr_src < ptr_src_end_simd && cnt_found < dst_len) {
        /* remaining dst */
        float buf_tmp_val[CACHE_LINE_NUMEL];
        uint32_t buf_tmp_idx[CACHE_LINE_NUMEL];
        uint32_t remaining_len = dst_len - cnt_found;
        assert(remaining_len < CACHE_LINE_NUMEL);

        while (ptr_src < ptr_src_end_simd) {
            __m256 vec_val0 = _mm256_loadu_ps(ptr_src);
            __m256 vec_val1 = _mm256_loadu_ps(ptr_src + SIMD_BLOCK);
            __m256 vec_val0_abs = _mm256_andnot_ps(mask_abs, vec_val0);
            __m256 vec_val1_abs = _mm256_andnot_ps(mask_abs, vec_val1);
            const float sum = hsum_float_avx(vec_val0_abs) + hsum_float_avx(vec_val1_abs);

            /* Here, we select first (dst_len - cnt_found) elements. However, this may not be optimal. Future room for optim. */

            if (sum >= threshold) {
                _mm256_storeu_ps(buf_tmp_val, vec_val0);
                _mm256_storeu_ps(buf_tmp_val + SIMD_BLOCK, vec_val1);
                _mm256_storeu_si256(reinterpret_cast<__m256i_u *>(buf_tmp_idx), vec_indices0);
                _mm256_storeu_si256(reinterpret_cast<__m256i_u *>(buf_tmp_idx + SIMD_BLOCK), vec_indices1);
                memcpy(ptr_dst_val, buf_tmp_val, sizeof(float) * remaining_len);
                memcpy(ptr_dst_idx, buf_tmp_idx, sizeof(uint32_t) * remaining_len);
                ptr_dst_idx += remaining_len;
                ptr_dst_val += remaining_len;
                cnt_found += remaining_len;
                assert(cnt_found == dst_len);
                ptr_src += remaining_len;
                break; // vec_indices0, vec_indices1 will be invalid afterwards!
            } else {
                vec_secondary_candidates.push_back(std::make_pair(sum, ptr_src - src.first));
            }

            vec_indices0 = _mm256_add_epi32(vec_indices0, vec_increment);
            vec_indices1 = _mm256_add_epi32(vec_indices1, vec_increment);
            ptr_src += CACHE_LINE_NUMEL;
        }
    }

    // std::cerr << "Stage 2: After manual, SRC = " << (ptr_src - src.first) << "/" << src.second << ", DST = " << cnt_found << "/" << dst_len << "\n";

    assert(ptr_src <= ptr_src_end);
    assert(cnt_found <= dst_len);

    if (ptr_src_end != ptr_src && cnt_found < dst_len) {
        assert(ptr_src_end - ptr_src < CACHE_LINE_NUMEL);
        uint32_t remaining_len = dst_len - cnt_found;
        uint32_t buf_tmp_idx[CACHE_LINE_NUMEL];
        
        float sum = 0;
        for (auto *p = ptr_src; p < ptr_src_end; ++p)
            sum += *p;
        
        /* Normalize threshold */
        if (sum * CACHE_LINE_NUMEL >= threshold * (ptr_src_end - ptr_src)) {
            uint32_t len_to_copy = std::min(remaining_len, static_cast<uint32_t>(ptr_src_end - ptr_src));
            memcpy(ptr_dst_val, ptr_src, sizeof(float) * len_to_copy);
            _mm256_storeu_si256(reinterpret_cast<__m256i_u *>(buf_tmp_idx), vec_indices0);
            _mm256_storeu_si256(reinterpret_cast<__m256i_u *>(buf_tmp_idx + SIMD_BLOCK), vec_indices1);
            memcpy(ptr_dst_idx, buf_tmp_idx, sizeof(uint32_t) * len_to_copy);
            ptr_dst_idx += len_to_copy;
            ptr_dst_val += len_to_copy;
            cnt_found += len_to_copy;
        } else {
            vec_secondary_candidates.push_back(std::make_pair(sum * CACHE_LINE_NUMEL / (ptr_src_end - ptr_src), ptr_src - src.first));
        }

        ptr_src = ptr_src_end;
    }

    assert(ptr_src <= ptr_src_end);
    assert(cnt_found <= dst_len);
    assert(ptr_src_end == ptr_src || cnt_found == dst_len);

#endif
    auto prev_thresh = threshold;

    // evaluate threshold is properly set, using AIMD
    if (cnt_found < dst_len) {
        // must drop less
        threshold = threshold * 0.99;
    } else {
        // must drop more
        threshold = threshold + threshold_inc;
    }
    // update the threshold
    
    {
        std::unique_lock<std::mutex> ul(threshold_map_mutex_);
        threshold_map_[ptr_key] = threshold;
        threshold_map_inc_[ptr_key] = threshold_inc;
    }

    if (cnt_found < dst_len) {
        
        // std::cerr << "Stage 3: After manual, SRC = " << (ptr_src - src.first) << "/" << src.second << ", DST = " << cnt_found << "/" << dst_len << std::endl;

        struct Compare{
            bool operator () (const std::pair<float, uint32_t>& lhs, const std::pair<float, uint32_t>& rhs) const{
                return lhs.first < rhs.first;
            }
        };
        
        std::priority_queue<std::pair<float, uint32_t>, std::vector<std::pair<float, uint32_t>>, Compare> q(Compare(), vec_secondary_candidates);

        
        while (cnt_found < dst_len) {
            assert(q.size() > 0);
            auto &selected = q.top();
            auto selected_idx = selected.second;
            uint32_t len_to_copy = std::min(16U, std::min(static_cast<uint32_t>(src.second - selected_idx), static_cast<uint32_t>(dst_len - cnt_found)));
            
            // std::cerr << "Selected sum = " << selected.first << ", prev_thresh=" << prev_thresh << ", idx=" << selected_idx << " - " << (selected_idx + len_to_copy) << "(" << len_to_copy << ")\n";

            memcpy(ptr_dst_val, src.first + selected_idx, sizeof(float) * len_to_copy);
            for (size_t i = 0; i < len_to_copy; i++) {
                ptr_dst_idx[i] = selected_idx + i + idx_offset;
            }
            
            cnt_found += len_to_copy;
            ptr_dst_idx += len_to_copy;
            ptr_dst_val += len_to_copy;

            q.pop();
        }
    }
    return cnt_found;
}

#if 0
// https://stackoverflow.com/questions/36932240/avx2-what-is-the-most-efficient-way-to-pack-left-based-on-a-mask
static inline std::pair<size_t, float> filter_threshold(float *__restrict__ dst, uint32_t *__restrict__ dst_index, const float *__restrict__ src,  size_t len, float v, int32_t idx_offset) {
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