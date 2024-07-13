#ifndef OPTIM_SPARSE_OPTIMIZER_H
#define OPTIM_SPARSE_OPTIMIZER_H

#include <stdint.h>
#include <stddef.h>
#include <string>
#include <torch/torch.h>
#include <torch/extension.h>

class SparseOptimizer {
protected:
    float m_lr;
    
    inline void checkValidity(torch::Tensor &param, torch::Tensor &grad_values, torch::Tensor &grad_indices) {
        TORCH_CHECK(param.is_contiguous());
        TORCH_CHECK(grad_values.is_contiguous());
        TORCH_CHECK(grad_indices.is_contiguous());

        TORCH_CHECK(param.dtype() == torch::Dtype::Float);
        TORCH_CHECK(grad_values.dtype() == torch::Dtype::Float);
        TORCH_CHECK(grad_indices.dtype() == torch::Dtype::Int);
        
        TORCH_CHECK(grad_indices.numel() == grad_values.numel());

        // TORCH_CHECK(!torch::any(torch::isnan(param)).item<bool>());
        // TORCH_CHECK(!torch::any(torch::isnan(grad_values)).item<bool>());
    }

public:
    SparseOptimizer() : m_lr(1e-3) {

    }
    
    float get_lr() const { return m_lr; }
    void set_lr(const float v) { m_lr = std::move(v); }

    
    virtual const std::string name() = 0;


    virtual void optimize(torch::Tensor &param, const std::string &name, torch::Tensor &grad_values, torch::Tensor &grad_indices) = 0;   
    virtual void optimize_raw(float *ptr_param, uint32_t param_len, const std::string &name, float *ptr_grad, uint32_t *ptr_idx, uint32_t grad_len) = 0;   

    virtual void configure(std::string &option_name, float option_value) = 0;
    virtual void configure(std::string &option_name, bool option_value) = 0;
};

#endif