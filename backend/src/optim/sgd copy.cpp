
#include "sgd.h"
#include <immintrin.h>

#define ASSUME_CACHE_FRIENDLY_INPUT 0
#define SIMD_OPTIMIZATION 1

inline bool is_aligned_64(void *ptr) {
    return reinterpret_cast<uintptr_t>(ptr) & 0x7 == 0;
}

static inline void _m256_i32scatter_ps(float* base_addr, __m256i indices, __m256 values) {
    base_addr[_mm256_extract_epi32(indices, 0)] = _mm256_cvtss_f32(_mm256_shuffle_ps(values, values, 0x00));
    base_addr[_mm256_extract_epi32(indices, 1)] = _mm256_cvtss_f32(_mm256_shuffle_ps(values, values, 0x55));
    base_addr[_mm256_extract_epi32(indices, 2)] = _mm256_cvtss_f32(_mm256_shuffle_ps(values, values, 0xaa));
    base_addr[_mm256_extract_epi32(indices, 3)] = _mm256_cvtss_f32(_mm256_shuffle_ps(values, values, 0xff));
    base_addr[_mm256_extract_epi32(indices, 4)] = _mm256_cvtss_f32(_mm256_shuffle_ps(values, _mm256_permute2f128_ps(values, values, 0x01), 0x00));
    base_addr[_mm256_extract_epi32(indices, 5)] = _mm256_cvtss_f32(_mm256_shuffle_ps(values, _mm256_permute2f128_ps(values, values, 0x01), 0x55));
    base_addr[_mm256_extract_epi32(indices, 6)] = _mm256_cvtss_f32(_mm256_shuffle_ps(values, _mm256_permute2f128_ps(values, values, 0x01), 0xaa));
    base_addr[_mm256_extract_epi32(indices, 7)] = _mm256_cvtss_f32(_mm256_shuffle_ps(values, _mm256_permute2f128_ps(values, values, 0x01), 0xff));
}

void SGD::optimize(torch::Tensor &param, const std::string &name, torch::Tensor &grad_values, torch::Tensor &grad_indices) {

    checkValidity(param, grad_values, grad_indices);

    auto ptr_param = reinterpret_cast<float *>(param.data_ptr());
    auto ptr_grad = reinterpret_cast<float *>(grad_values.data_ptr());
    auto ptr_idx = reinterpret_cast<uint32_t *>(grad_indices.data_ptr());

    optimize_raw(ptr_param, param.numel(), name, ptr_grad, ptr_idx, grad_values.numel());
}

void SGD::optimize_raw(float *ptr_param, uint32_t param_len, const std::string &name, float *ptr_grad, uint32_t *ptr_idx, uint32_t grad_len) {
    bool first = false;
    auto tensor_len = param_len;
    const size_t gradient_len = grad_len;

    // std::cout << "name - " << name << " " << m_dampening << std::endl;
    
    {
        std::unique_lock<std::mutex> ul(m_optim_state_b_mutex_);
        if (m_momentum != 0 && m_optim_state_b.find(name) == m_optim_state_b.end()) {
            m_optim_state_b[name] = std::make_unique<float[]>(tensor_len);
            memset(m_optim_state_b[name].get(), 0, tensor_len * sizeof(float));
            m_optim_state_last[name] = std::make_unique<uint32_t []>(tensor_len);
            memset(m_optim_state_last[name].get(), 0, tensor_len * sizeof(uint32_t));
            first = true;
        }
    }
    const double lr = m_maximize ? -m_lr : m_lr; // need double to retain precision
    
    auto ptr_arr_b = m_momentum ? m_optim_state_b[name].get() : nullptr;
    auto ptr_arr_last = m_smart_momentum ? m_optim_state_last[name].get() : nullptr;

#if (SIMD_OPTIMIZATION) && (__AVX512F__)
    /* fast path */
    if (m_momentum != 0 && m_weight_decay == 0 && !m_nestrov) {
        size_t i;
        __m256 vec_lr = _mm256_set1_ps(-lr);
        
        if (__glibc_unlikely(first)) {
            for (i = 0; i + 8 < gradient_len; i += 8) {
                /* prefetch for cache-friendliness */
                uint32_t idx_first = ptr_idx[i];
                _mm_prefetch(&ptr_param[idx_first], _MM_HINT_NTA);

                __m256i vec_ptr_idx = _mm256_loadu_si256(reinterpret_cast<__m256i *>(&ptr_idx[i]));
                __m256 vec_grad = _mm256_loadu_ps(&ptr_grad[i]);
#if ASSUME_CACHE_FRIENDLY_INPUT
                __m512 vec_param_x = _mm256_loadu_ps(&ptr_param[idx_first]); 
#else
                __m256 vec_param_x = _mm256_i32gather_ps(ptr_param, vec_ptr_idx, 4); 
#endif
                /* ptr_arr_b[idx] = grad; */
                _m256_i32scatter_ps(ptr_arr_b, vec_ptr_idx, vec_grad); /* no intrinsics in 256-bit */
                
                /* x - lr * grad (single-precision implementation) */
                __m256 vec_result = _mm256_fmadd_ps(vec_lr, vec_grad, vec_param_x);

                // ptr_param[idx] = x - lr * grad;
                
#if ASSUME_CACHE_FRIENDLY_INPUT
                _mm512_storeu_ps(&ptr_param[idx_first], vec_result);
#else
                _m256_i32scatter_ps(ptr_param, vec_ptr_idx, vec_result); /* no intrinsics in 256-bit */
#endif

            }

            for (; i < gradient_len; i++) {
                const uint32_t idx = ptr_idx[i];
                float x = ptr_param[idx];
                float grad = ptr_grad[i];
                ptr_arr_b[idx] = grad;
                ptr_param[idx] = x - lr * grad;
            }  
        } else {
            __m256 vec_momentum = _mm256_set1_ps(m_momentum);
            __m256 vec_dampening = _mm256_set1_ps(1-m_dampening);
            const size_t simd_boundary = gradient_len - 64;
            i = 0;

            if (gradient_len >= 64) {
                for (; i < simd_boundary; i += 32) {
                    __m256i vec_ptr_idx0 = _mm256_loadu_si256(reinterpret_cast<__m256i *>(&ptr_idx[i]));
                    __m256i vec_ptr_idx1 = _mm256_loadu_si256(reinterpret_cast<__m256i *>(&ptr_idx[i+8]));
                    __m256i vec_ptr_idx2 = _mm256_loadu_si256(reinterpret_cast<__m256i *>(&ptr_idx[i+16]));
                    __m256i vec_ptr_idx3 = _mm256_loadu_si256(reinterpret_cast<__m256i *>(&ptr_idx[i+24]));

                    uint32_t idx_first0 = _mm256_extract_epi32(vec_ptr_idx0, 0);
                    uint32_t idx_first1 = _mm256_extract_epi32(vec_ptr_idx1, 0);
                    uint32_t idx_first2 = _mm256_extract_epi32(vec_ptr_idx2, 0);
                    uint32_t idx_first3 = _mm256_extract_epi32(vec_ptr_idx3, 0);
                    
                    /* prefetch for cache-friendliness */
                    _mm_prefetch(&ptr_arr_b[idx_first0], _MM_HINT_NTA);
                    _mm_prefetch(&ptr_arr_b[idx_first1], _MM_HINT_NTA);
                    _mm_prefetch(&ptr_arr_b[idx_first2], _MM_HINT_NTA);
                    _mm_prefetch(&ptr_arr_b[idx_first3], _MM_HINT_NTA);
                    _mm_prefetch(&ptr_param[idx_first0], _MM_HINT_NTA);
                    _mm_prefetch(&ptr_param[idx_first1], _MM_HINT_NTA);
                    _mm_prefetch(&ptr_param[idx_first2], _MM_HINT_NTA);
                    _mm_prefetch(&ptr_param[idx_first3], _MM_HINT_NTA);


                    __m256 vec_grad0 = _mm256_loadu_ps(&ptr_grad[i]);
                    __m256 vec_grad1 = _mm256_loadu_ps(&ptr_grad[i+8]);
                    __m256 vec_grad2 = _mm256_loadu_ps(&ptr_grad[i+16]);
                    __m256 vec_grad3 = _mm256_loadu_ps(&ptr_grad[i+24]);

#if ASSUME_CACHE_FRIENDLY_INPUT
                    /* has further optimization chance, if guaranteed sequential */
                    __m512 vec_arr_b0 = _mm512_loadu_ps(&ptr_arr_b[idx_first0]); 
                    __m512 vec_arr_b1 = _mm512_loadu_ps(&ptr_arr_b[idx_first1]); 
                    __m512 vec_arr_b2 = _mm512_loadu_ps(&ptr_arr_b[idx_first2]); 
                    __m512 vec_arr_b3 = _mm512_loadu_ps(&ptr_arr_b[idx_first3]); 
                    
                    __m512 vec_param_x0 = _mm512_loadu_ps(&ptr_param[idx_first0]); 
                    __m512 vec_param_x1 = _mm512_loadu_ps(&ptr_param[idx_first1]); 
                    __m512 vec_param_x2 = _mm512_loadu_ps(&ptr_param[idx_first2]); 
                    __m512 vec_param_x3 = _mm512_loadu_ps(&ptr_param[idx_first3]); 
#else
                    __m256 vec_arr_b0 = _mm256_i32gather_ps(ptr_arr_b, vec_ptr_idx0, 4); 
                    __m256 vec_arr_b1 = _mm256_i32gather_ps(ptr_arr_b, vec_ptr_idx1, 4); 
                    __m256 vec_arr_b2 = _mm256_i32gather_ps(ptr_arr_b, vec_ptr_idx2, 4); 
                    __m256 vec_arr_b3 = _mm256_i32gather_ps(ptr_arr_b, vec_ptr_idx3, 4); 

                    __m256 vec_param_x0 = _mm256_i32gather_ps(ptr_param, vec_ptr_idx0, 4); 
                    __m256 vec_param_x1 = _mm256_i32gather_ps(ptr_param, vec_ptr_idx1, 4); 
                    __m256 vec_param_x2 = _mm256_i32gather_ps(ptr_param, vec_ptr_idx2, 4); 
                    __m256 vec_param_x3 = _mm256_i32gather_ps(ptr_param, vec_ptr_idx3, 4); 
#endif
                    
                    /*  grad = ptr_arr_b[idx] * m_momentum + (1 - m_dampening) * grad */
                    vec_grad0 = _mm256_mul_ps(vec_grad0, vec_dampening);
                    vec_grad1 = _mm256_mul_ps(vec_grad1, vec_dampening);
                    vec_grad2 = _mm256_mul_ps(vec_grad2, vec_dampening);
                    vec_grad3 = _mm256_mul_ps(vec_grad3, vec_dampening);

                    vec_grad0 = _mm256_fmadd_ps(vec_momentum, vec_arr_b0, vec_grad0);
                    vec_grad1 = _mm256_fmadd_ps(vec_momentum, vec_arr_b1, vec_grad1);
                    vec_grad2 = _mm256_fmadd_ps(vec_momentum, vec_arr_b2, vec_grad2);
                    vec_grad3 = _mm256_fmadd_ps(vec_momentum, vec_arr_b3, vec_grad3);

#if ASSUME_CACHE_FRIENDLY_INPUT
                    /* ptr_arr_b[idx] = grad */
                    _mm512_storeu_ps(&ptr_arr_b[idx_first0], vec_grad0);
                    _mm512_storeu_ps(&ptr_arr_b[idx_first1], vec_grad1);
                    _mm512_storeu_ps(&ptr_arr_b[idx_first2], vec_grad2);
                    _mm512_storeu_ps(&ptr_arr_b[idx_first3], vec_grad3);
#else
                    /* ptr_arr_b[idx] = grad */
                    _mm256_i32scatter_ps(ptr_arr_b, vec_ptr_idx0, vec_grad0, 4);
                    _mm256_i32scatter_ps(ptr_arr_b, vec_ptr_idx1, vec_grad1, 4);
                    _mm256_i32scatter_ps(ptr_arr_b, vec_ptr_idx2, vec_grad2, 4);
                    _mm256_i32scatter_ps(ptr_arr_b, vec_ptr_idx3, vec_grad3, 4);
#endif

                    /*  x - lr * grad */
                    __m256 vec_result0 = _mm256_fmadd_ps(vec_lr, vec_grad0, vec_param_x0);
                    __m256 vec_result1 = _mm256_fmadd_ps(vec_lr, vec_grad1, vec_param_x1);
                    __m256 vec_result2 = _mm256_fmadd_ps(vec_lr, vec_grad2, vec_param_x2);
                    __m256 vec_result3 = _mm256_fmadd_ps(vec_lr, vec_grad3, vec_param_x3);

                    // ptr_param[idx] = x - lr * grad;
                    
#if ASSUME_CACHE_FRIENDLY_INPUT
                    _mm512_storeu_ps(&ptr_param[idx_first0], vec_result0);
                    _mm512_storeu_ps(&ptr_param[idx_first1], vec_result1);
                    _mm512_storeu_ps(&ptr_param[idx_first2], vec_result2);
                    _mm512_storeu_ps(&ptr_param[idx_first3], vec_result3);
#else
                    _mm256_i32scatter_ps(ptr_param, vec_ptr_idx0, vec_result0, 4);
                    _mm256_i32scatter_ps(ptr_param, vec_ptr_idx1, vec_result1, 4);
                    _mm256_i32scatter_ps(ptr_param, vec_ptr_idx2, vec_result2, 4);
                    _mm256_i32scatter_ps(ptr_param, vec_ptr_idx3, vec_result3, 4);
#endif
                }

            }


            for (; i < gradient_len; i++) {
                const uint32_t idx = ptr_idx[i];
                float x = ptr_param[idx];
                float grad = ptr_grad[i];
                grad = ptr_arr_b[idx] * m_momentum + (1 - m_dampening) * grad;
                ptr_arr_b[idx] = grad;
                ptr_param[idx] = x - lr * grad;
            }  
        }
        return;
    }
#endif

    /**
     * Implemented per https://pytorch.org/docs/stable/generated/torch.optim.SGD.html
    */

    for (size_t i = 0; i < gradient_len; i++) {
        const uint32_t idx = ptr_idx[i];
        float x = ptr_param[idx];
        float grad = ptr_grad[i];
        float momentum = m_smart_momentum ? 
            (m_iter == 0 ? m_momentum : std::pow(m_momentum, m_iter - ptr_arr_last[idx])) : 
            m_momentum;

        if (!(momentum < 1 && momentum >= 0)) {
            std::cerr << "Invalid momentum " << momentum << " (m_iter = " << m_iter << ", ptr_arr_last[idx] = "<< ptr_arr_last[idx] << ")" << std::endl;
            assert(momentum < 1 && momentum >= 0);
        }

        if (m_weight_decay != 0) {
            grad += m_weight_decay * x;
        }

        if (m_momentum != 0) {
            float optim_b = 0;

            if (!first) {
                optim_b = ptr_arr_b[idx] * momentum + (1 - m_dampening) * grad;
            } else {
                optim_b = grad;
            }

            if (m_nestrov) {
                grad += momentum * optim_b;
            } else {
                grad = optim_b;
            }

            ptr_arr_b[idx] = optim_b;
        }

        ptr_param[idx] = x - lr * grad;

        if (m_smart_momentum)
            ptr_arr_last[idx] = m_iter;
    }

    m_iter++;
}


void SGD::configure(std::string &option_name, float option_value) {
    if (option_name == "lr") {
        m_lr = option_value;
        return;
    } else if (option_name == "momentum") {
        m_momentum = option_value;
        return;
    } else if (option_name == "weight_decay") {
        m_weight_decay = option_value;
        return;
    } else if (option_name == "dampening") {
        m_dampening = option_value;
        return;
    }
    
    throw std::runtime_error("NO OPTION");
}

void SGD::configure(std::string &option_name, bool option_value) {
    if (option_name == "nestrov") {
        m_nestrov = option_value;
        return;
    } else if (option_name == "maximize") {
        m_maximize = option_value;
        return;
    } else if (option_name == "smart_momentum") {
        m_smart_momentum = option_value;
        return;
    }

    throw std::runtime_error("NO OPTION");
}