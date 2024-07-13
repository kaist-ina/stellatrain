#include "adam.h"
#include <cmath>
#include <stdexcept>


void Adam::optimize(torch::Tensor &param, const std::string &name, torch::Tensor &grad_values, torch::Tensor &grad_indices) {
    checkValidity(param, grad_values, grad_indices);

    optimize_raw(
        reinterpret_cast<float *>(param.data_ptr()),
        param.numel(),
        name,
        reinterpret_cast<float *>(grad_values.data_ptr()),
        reinterpret_cast<uint32_t *>(grad_indices.data_ptr()),
        grad_values.numel()
    );
}

void Adam::optimize_raw(float *ptr_param, uint32_t param_len, const std::string &name, float *ptr_grad, uint32_t *ptr_idx, uint32_t grad_len) {        

    /**
     * https://pytorch.org/docs/stable/generated/torch.optim.Adam.html
    */

    auto tensor_len = param_len;
    auto gradient_len = grad_len;

    {
        std::unique_lock<std::mutex> ul(m_optim_state_mutex_);
        if (m_optim_state_m.find(name) == m_optim_state_m.end()) {
            m_optim_state_m[name] = std::make_unique<float[]>(tensor_len);
            m_optim_state_v[name] = std::make_unique<float[]>(tensor_len);
            m_optim_state_vmax[name] = 0;
            m_optim_state_tick[name] = 1;
        }
    }

    auto optim_state_m = m_optim_state_m[name].get();
    auto optim_state_v = m_optim_state_v[name].get();
    auto optim_state_vmax = m_optim_state_vmax[name];
    auto tick = m_optim_state_tick[name];
    auto m_b1_pow = std::pow(m_b1, tick);
    auto m_b2_pow = std::pow(m_b2, tick);
    const double lr = m_lr;
    
    {   
        std::unique_lock<std::mutex> ul(m_optim_state_mutex_);
        for (size_t i = 0; i < gradient_len; i++) {
            const uint32_t idx = ptr_idx[i];
            float grad = ptr_grad[i];
            float x = ptr_param[idx];
            float m = optim_state_m[idx];
            float v = optim_state_v[idx];
            float vmax = optim_state_vmax;

            if (m_maximize) {
                grad = -grad;    
            }

            if (m_weight_decay != 0) {
                grad += m_weight_decay * x;
            }

            double mt = m_b1 * m + (1 - m_b1) * grad;
            double vt = m_b2 * v + (1 - m_b2) * grad * grad;

            double mt_hat = mt / (1 - m_b1_pow);
            double vt_hat = vt / (1 - m_b2_pow);

            if (m_amsgrad) {
                optim_state_vmax = std::max((double)optim_state_vmax, vt_hat);
                ptr_param[idx] = x - lr * mt_hat / (std::sqrt(optim_state_vmax) + m_eps);
            } else {
                ptr_param[idx] = x - lr * mt_hat / (std::sqrt(vt_hat) + m_eps);
            }
            
            // save state
            optim_state_m[idx] = mt;
            optim_state_v[idx] = vt;
        }

        m_optim_state_vmax[name] = optim_state_vmax;
        m_optim_state_tick[name]++;
    }
    
   
}



void Adam::configure(std::string &option_name, float option_value) {
    if (option_name == "lr") {
        m_lr = option_value;
        return;
    } else if (option_name == "b1") {
        m_b1 = option_value;
        return;
    } else if (option_name == "b2") {
        m_b2 = option_value;
        return;
    } else if (option_name == "weight_decay") {
        m_weight_decay = option_value;
        return;
    } else if (option_name == "eps") {
        m_eps = option_value;
        return;
    }
    
    throw std::runtime_error("NO OPTION");
}

void Adam::configure(std::string &option_name, bool option_value) {
    if (option_name == "amsgrad") {
        m_amsgrad = option_value;
        return;
    } else if (option_name == "maximize") {
        m_maximize = option_value;
        return;
    }

    throw std::runtime_error("NO OPTION");
}