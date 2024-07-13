#ifndef OPTIM_ADAM_H
#define OPTIM_ADAM_H
#include "sparse_optimizer.h"
#include <utility>
#include <unordered_map>
#include <memory>
#include <string>
#include <mutex>

class Adam : public SparseOptimizer {

private:
    float m_b1, m_b2, m_weight_decay, m_eps;
    bool m_amsgrad, m_maximize;
    std::unordered_map<std::string, std::unique_ptr<float[]>> m_optim_state_m;
    std::unordered_map<std::string, std::unique_ptr<float[]>> m_optim_state_v;
    std::unordered_map<std::string, float> m_optim_state_vmax;
    std::unordered_map<std::string, uint32_t> m_optim_state_tick;
    std::mutex m_optim_state_mutex_;

public:
    Adam() : m_b1(0.9), m_b2(0.999), m_weight_decay(0), m_eps(1e-8), m_amsgrad(false), m_maximize(false) {

    }

    ~Adam() {
        m_optim_state_m.clear();
        m_optim_state_v.clear();
        m_optim_state_vmax.clear();
        m_optim_state_tick.clear();
    }

    virtual const std::string name() {
        return std::string("Adam");
    };

    float get_b1() const { return m_b1; }
    void set_b1(const float v) { m_b1 = std::move(v); }

    float get_b2() const { return m_b1; }
    void set_b2(const float v) { m_b2 = std::move(v); }

    float get_weight_decay() const { return m_weight_decay; }
    void set_weight_decay(const float v) { m_weight_decay = std::move(v); }

    
    float get_eps() const { return m_eps; }
    void set_eps(const float v) { m_eps = std::move(v); }

    virtual void optimize(torch::Tensor &param, const std::string &name, torch::Tensor &grad_values, torch::Tensor &grad_indices);
    virtual void optimize_raw(float *ptr_param, uint32_t param_len, const std::string &name, float *ptr_grad, uint32_t *ptr_idx, uint32_t grad_len);

    
    virtual void configure(std::string &option_name, float option_value);
    virtual void configure(std::string &option_name, bool option_value);
};

#endif