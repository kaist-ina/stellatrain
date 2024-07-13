#ifndef _MISC_ARRAY_UTIL_H
#define _MISC_ARRAY_UTIL_H
#include <immintrin.h>


/**
 * @brief SIMD optimized addition of two given arrays. Add \a dst and \a src and save to  \a dst`.
 * @param dst destination.
 * @param src source.
 * @param length The number of **elements** in the array, not bytes.
*/
static inline void add_arrays(float* dst, const float* src, std::size_t length) {
    const float *end = src + length;

#if __AVX512F__
    /* AVX-512F */
    for(; src + 64 < end; src += 64, dst += 64) {
        __m512 vec_dst0 = _mm512_loadu_ps(dst + 0);
        __m512 vec_dst1 = _mm512_loadu_ps(dst + 16);
        __m512 vec_dst2 = _mm512_loadu_ps(dst + 32);
        __m512 vec_dst3 = _mm512_loadu_ps(dst + 48);
        __m512 vec_src0 = _mm512_loadu_ps(src + 0);
        __m512 vec_src1 = _mm512_loadu_ps(src + 16);
        __m512 vec_src2 = _mm512_loadu_ps(src + 32);
        __m512 vec_src3 = _mm512_loadu_ps(src + 48);
        _mm512_storeu_ps(dst + 0,  _mm512_add_ps(vec_dst0, vec_src0));
        _mm512_storeu_ps(dst + 16,  _mm512_add_ps(vec_dst1, vec_src1));
        _mm512_storeu_ps(dst + 32, _mm512_add_ps(vec_dst2, vec_src2));
        _mm512_storeu_ps(dst + 48, _mm512_add_ps(vec_dst3, vec_src3));
    }
#endif

    /* AVX */
    for(; src + 32 < end; src += 32, dst += 32) {
        __m256 vec_dst0 = _mm256_loadu_ps(dst + 0);
        __m256 vec_dst1 = _mm256_loadu_ps(dst + 8);
        __m256 vec_dst2 = _mm256_loadu_ps(dst + 16);
        __m256 vec_dst3 = _mm256_loadu_ps(dst + 24);
        __m256 vec_src0 = _mm256_loadu_ps(src + 0);
        __m256 vec_src1 = _mm256_loadu_ps(src + 8);
        __m256 vec_src2 = _mm256_loadu_ps(src + 16);
        __m256 vec_src3 = _mm256_loadu_ps(src + 24);
        _mm256_storeu_ps(dst + 0,  _mm256_add_ps(vec_dst0, vec_src0));
        _mm256_storeu_ps(dst + 8,  _mm256_add_ps(vec_dst1, vec_src1));
        _mm256_storeu_ps(dst + 16, _mm256_add_ps(vec_dst2, vec_src2));
        _mm256_storeu_ps(dst + 24, _mm256_add_ps(vec_dst3, vec_src3));
    }

    for(; src < end; src++, dst++) {
        *dst += *src;
    }
}


#endif