#include "sgd_naive.h"

void SGDNaive::optimize(torch::Tensor &param, const std::string &name, torch::Tensor &grad_values, torch::Tensor &grad_indices) {

    // checkValidity(param, grad_values, grad_indices);

    at::Tensor dense_grad = torch::zeros_like(param);
    dense_grad.index_put_({grad_indices}, grad_values, true);

    bool first = false;
    auto tensor_len = param.numel();

    {
        std::unique_lock<std::mutex> ul(m_optim_state_b_mutex_);
        if (m_momentum != 0 && m_optim_state_b.find(name) == m_optim_state_b.end()) {
            m_optim_state_b[name] = torch::zeros({tensor_len});
            m_optim_state_last[name] = torch::zeros({tensor_len}, at::TensorOptions().dtype(at::kInt));
            first = true;
        }
    }

    
    const double lr = m_maximize ? -m_lr : m_lr; // need double to retain precision

    if (m_weight_decay != 0) {
        dense_grad += m_weight_decay * param;
    }

    at::Tensor optim_b;
    if (m_momentum) {
        if (!first) {
            optim_b = m_optim_state_b[name] * m_momentum + (1 - m_dampening) * dense_grad;
        } else {
            optim_b = dense_grad;
        }
    }

    if (m_nestrov) {
        dense_grad += m_momentum * optim_b;
    } else {
        dense_grad = optim_b;
    }

    m_optim_state_b[name] = optim_b;

    param = param - lr * dense_grad;
    // optimize_raw(ptr_param, param.numel(), name, ptr_grad, ptr_idx, grad_values.numel());
}

void SGDNaive::optimize_raw(float *ptr_param, uint32_t param_len, const std::string &name, float *ptr_grad, uint32_t *ptr_idx, uint32_t grad_len) {
    
    at::Tensor param = torch::from_blob(ptr_param, {static_cast<long>(param_len)}, at::kFloat);
    at::Tensor grad_values = torch::from_blob(ptr_grad, {static_cast<long>(grad_len)}, at::kFloat);
    at::Tensor grad_indices = torch::from_blob(ptr_idx, {static_cast<long>(grad_len)}, at::kInt).to(at::kLong);

    optimize(param, name, grad_values, grad_indices);
}


void SGDNaive::configure(std::string &option_name, bool option_value) {
    if (option_name == "nestrov") {
        m_nestrov = option_value;
        return;
    } else if (option_name == "maximize") {
        m_maximize = option_value;
        return;
    }

    throw std::runtime_error("NO OPTION");
}

void SGDNaive::configure(std::string &option_name, float option_value) {
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
