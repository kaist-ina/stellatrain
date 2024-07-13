#ifndef _ENGINE_TELEMERTY_SERVER_H_
#define _ENGINE_TELEMERTY_SERVER_H_

#include <string>
#include <vector>
#include <mutex>
#include <thread>
#include <memory>
#include <zmq.hpp>
#include "message.h"

struct GlobalTelemetry {
    size_t num_nodes;
    size_t num_gpus;
    std::vector<size_t> param_counts_per_layer;
    bool grad_fp16;
    bool grad_idx_u16;

    GlobalTelemetry() = default;
    GlobalTelemetry(size_t num_nodes, size_t num_gpus, std::vector<size_t> param_counts_per_layer, bool grad_fp16, bool grad_idx_u16) :
        num_nodes(num_nodes), num_gpus(num_gpus), param_counts_per_layer(param_counts_per_layer), grad_fp16(grad_fp16), grad_idx_u16(grad_idx_u16) {}
};

class TelemetryServer {
    zmq::context_t zmq_ctx_;
    zmq::socket_t sock_;
    std::mutex mutex_;
    std::unique_ptr<std::thread> python_launch_thread_;

public:
    TelemetryServer(const std::string telemetry_uri);
    ~TelemetryServer();

    void send_global_telemetry(GlobalTelemetry telemetry);

    int python_launch_thread_main();
};

#endif // _ENGINE_BATCH_RATE_ALLOC_H_
