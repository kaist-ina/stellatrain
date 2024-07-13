#ifndef OPTIM_SGD_H
#define OPTIM_SGD_H
#include "sparse_optimizer.h"
#include <utility>
#include <unordered_map>
#include <memory>
#include <string>
#include <mutex>

class SGD : public SparseOptimizer {

private:
    float m_momentum, m_weight_decay, m_dampening;
    bool m_nestrov, m_maximize, m_smart_momentum;
    std::unordered_map<std::string, std::unique_ptr<float[]>> m_optim_state_b;
    std::unordered_map<std::string, std::unique_ptr<uint32_t []>> m_optim_state_last;
    std::mutex m_optim_state_b_mutex_;
    uint32_t m_iter;

public:
    SGD() : m_momentum(0), m_weight_decay(0), m_dampening(0), m_nestrov(false), m_maximize(false), m_iter(0), m_smart_momentum(false) {

    }

    ~SGD() {
        m_optim_state_b.clear();
    }
    
    virtual const std::string name() {
        return std::string("SGD");
    };

    float get_lr() const { return m_lr; }
    void set_lr(const float v) { m_lr = std::move(v); }

    float get_momentum() const { return m_momentum; }
    void set_momentum(const float v) { m_momentum = std::move(v); }

    bool get_smart_momentum() const { return m_smart_momentum; }
    void set_smart_momentum(const bool v) { m_smart_momentum = v; }
    
    virtual void optimize(torch::Tensor &param, const std::string &name, torch::Tensor &grad_values, torch::Tensor &grad_indices);
    virtual void optimize_raw(float *ptr_param, uint32_t param_len, const std::string &name, float *ptr_grad, uint32_t *ptr_idx, uint32_t grad_len);

    
    virtual void configure(std::string &option_name, float option_value);
    virtual void configure(std::string &option_name, bool option_value);

};

#endif