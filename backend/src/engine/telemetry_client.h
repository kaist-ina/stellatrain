#ifndef _ENGINE_TELEMETRY_CLIENT_H
#define _ENGINE_TELEMETRY_CLIENT_H

#include <zmq.hpp>
#include <cstdint>
#include <string>
#include <mutex>
#include <condition_variable>
#include <memory>
#include <thread>
#include <functional>
#include <queue>
#include "config.h"
#include "message.h"

#ifndef PROTOBUF_USE_DLLS
#define PROTOBUF_USE_DLLS
#endif

class TelemetryClient {
    
private:
    uint32_t session_id_ = 0;
    zmq::context_t zmq_ctx_;
    zmq::socket_t sock_;

    size_t gpu_id_;

    size_t telemetry_message_count_ = 0;

    std::unique_ptr<std::thread> instruction_handler_thread_;
    std::condition_variable cond_;
    std::mutex mutex_;
    std::queue<std::pair<std::string, std::function<void(int iter_effective_since, TrainConfig config)>>> queue_;
    bool finished_ = false;
    
    void comm_handler_main();

public:
    TelemetryClient(const std::string uri, size_t gpu_id, const std::string gpu_name, int rank, int local_rank);
    ~TelemetryClient();

    /** Report Telemetry to Telemetry Server. Non-blocking call */
    void report(int uiter, TrainConfig config, TrainMeasurement measurement, std::function<void(int iter_effective_since, TrainConfig config)> callback);

};


#endif
