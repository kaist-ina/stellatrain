#include <iostream>
#include <cstdlib>
#include <csignal>
#include <cassert>
#include <pthread.h>
#include "telemetry_server.h"
#include "src/proto/message.pb.h"


TelemetryServer::TelemetryServer(const std::string telemetry_uri) {
    GOOGLE_PROTOBUF_VERIFY_VERSION;
    python_launch_thread_ = std::make_unique<std::thread>(&TelemetryServer::python_launch_thread_main, this);
    
     // connect zmq socket
    sock_ = zmq::socket_t(zmq_ctx_, ZMQ_REQ);
    sock_.connect(telemetry_uri);

    // send msg
    fasterdp::Request msg;
    msg.set_type(fasterdp::Request::INIT);
    const auto &serialized = msg.SerializeAsString();
    
    {
        std::unique_lock<std::mutex> lock(mutex_);
        sock_.send(zmq::buffer(serialized.c_str(), serialized.length()));

        // dummy recv
        zmq::message_t recv_msg;
        auto result = sock_.recv(recv_msg);
        assert(result.has_value());
    }
}


TelemetryServer::~TelemetryServer() {

    fasterdp::Request msg;
    msg.set_type(fasterdp::Request::TERMINATE);
    const auto &serialized = msg.SerializeAsString();
    
    {
        std::unique_lock<std::mutex> lock(mutex_);
        sock_.send(zmq::buffer(serialized.c_str(), serialized.length()));

        // dummy recv
        zmq::message_t recv_msg;
        auto result = sock_.recv(recv_msg);
        assert(result.has_value());
        sock_.close();
    }

    /* wait for python thread to terminate */
    python_launch_thread_->join();
    std::cerr << "Python thread terminated" << std::endl;
}

int TelemetryServer::python_launch_thread_main() {
    sigset_t mask;
    sigemptyset (&mask);
    sigaddset (&mask, SIGTERM);
    sigaddset (&mask, SIGHUP);
    pthread_sigmask(SIG_BLOCK, &mask, NULL);
    pthread_setname_np(pthread_self(), "PythonLaunch");
    
    /* Quick and Dirty, need to update later */
    int ret = std::system("python ../src/engine/batch_rate_alloc.py ");

    if (ret != 0 && ret != 130 && ret != 2) {
        std::cerr << "Python thread terminated abnormally with status code " << ret << std::endl;
        abort();
    }
    return ret;
}

void TelemetryServer::send_global_telemetry(GlobalTelemetry telemetry) {
    fasterdp::Request msg;
    msg.set_type(fasterdp::Request::GLOBAL_TELEMETRY_INIT);
    msg.mutable_global_telemetry_init()->set_num_nodes(telemetry.num_nodes);
    msg.mutable_global_telemetry_init()->set_num_gpus(telemetry.num_gpus);
    msg.mutable_global_telemetry_init()->set_grad_fp16(telemetry.grad_fp16);
    msg.mutable_global_telemetry_init()->set_grad_idx_u16(telemetry.grad_idx_u16);
    for (auto param_count : telemetry.param_counts_per_layer) {
        msg.mutable_global_telemetry_init()->add_param_counts(param_count);
    }
    const auto &serialized = msg.SerializeAsString();
    
    {
        std::unique_lock<std::mutex> lock(mutex_);
        sock_.send(zmq::buffer(serialized.c_str(), serialized.length()));

        // dummy recv
        zmq::message_t recv_msg;
        auto result = sock_.recv(recv_msg);
        assert(result.has_value());
    }
}
