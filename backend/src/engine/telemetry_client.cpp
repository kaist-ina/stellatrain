#include "telemetry_client.h"
#include "src/proto/message.pb.h"
#include <cassert>

TelemetryClient::TelemetryClient(const std::string uri, size_t gpu_id, const std::string gpu_name, int rank, int local_rank) : gpu_id_(gpu_id) {
    GOOGLE_PROTOBUF_VERIFY_VERSION;

    // connect zmq socket
    sock_ = zmq::socket_t(zmq_ctx_, ZMQ_REQ);
    sock_.connect(uri);
    

    // send telemetry init msg
    fasterdp::Request msg;
    msg.set_type(fasterdp::Request::LOCAL_TELEMETRY_INIT);
    msg.mutable_local_telemetry_init()->set_client_id(gpu_id);
    msg.mutable_local_telemetry_init()->set_rank(rank);
    msg.mutable_local_telemetry_init()->set_local_rank(local_rank);
    msg.mutable_local_telemetry_init()->set_gpu_name(gpu_name);

    const auto &serialized = msg.SerializeAsString();
    sock_.send(zmq::buffer(serialized.c_str(), serialized.length()));

    // blocking wait
    zmq::message_t recv_msg;
    sock_.recv(recv_msg);

    instruction_handler_thread_ = std::make_unique<std::thread>(&TelemetryClient::comm_handler_main, this);
}
TelemetryClient::~TelemetryClient() {
    finished_ = true;
    if (instruction_handler_thread_ != nullptr) {
        cond_.notify_all();
        instruction_handler_thread_->join();
    }
    google::protobuf::ShutdownProtobufLibrary(); 
}

void TelemetryClient::report(int uiter, TrainConfig config, TrainMeasurement measurement, std::function<void(int iter_effective_since, TrainConfig config)> callback) {

    fasterdp::Request msg;
    msg.set_type(fasterdp::Request::LOCAL_TELEMETRY);

    msg.mutable_telemetry()->set_client_id(gpu_id_);
    msg.mutable_telemetry()->set_message_id(telemetry_message_count_);
    msg.mutable_telemetry()->set_uiter_count(uiter);

    msg.mutable_telemetry()->mutable_config()->set_compression_rate(config.compression_rate);
    msg.mutable_telemetry()->mutable_config()->set_gradient_accumulation(config.gradient_accumulation);
    msg.mutable_telemetry()->mutable_config()->set_ubatch_size(config.ubatch_size);

    msg.mutable_telemetry()->mutable_measurements()->set_grad_send_thpt_mbps(measurement.grad_send_thpt_mbps);
    msg.mutable_telemetry()->mutable_measurements()->set_grad_recv_thpt_mbps(measurement.grad_recv_thpt_mbps);
    msg.mutable_telemetry()->mutable_measurements()->set_fwd_delay_ms(measurement.fwd_delay_ms);
    msg.mutable_telemetry()->mutable_measurements()->set_bwd_delay_ms(measurement.bwd_delay_ms);
    msg.mutable_telemetry()->mutable_measurements()->set_bwd_fwd_gap_ms(measurement.bwd_fwd_gap_ms);
    msg.mutable_telemetry()->mutable_measurements()->set_intra_fwd_gap_ms(measurement.intra_fwd_gap_ms);


    // std::cerr << "Sending telemetry message " << uiter << std::endl;

    const auto &serialized = msg.SerializeAsString();
    
    {
        std::unique_lock<std::mutex> ul(mutex_);
        queue_.emplace(std::make_pair(serialized, callback));
    }
    cond_.notify_one();
}


void TelemetryClient::comm_handler_main() {
    while (!finished_) {
        std::unique_lock<std::mutex> ul(mutex_);
        cond_.wait(ul, [this] () { return finished_ || queue_.size() > 0; });

        if (queue_.size() == 0) {
            continue;
        }

        assert(queue_.size() > 0);

        auto pair = queue_.front();
        queue_.pop();
        ul.unlock();

        sock_.send(zmq::buffer(pair.first.c_str(), pair.first.length()));

        zmq::message_t recv_msg;
        auto result = sock_.recv(recv_msg);
        assert(result.has_value());

        fasterdp::Response resp;
        resp.ParseFromArray(recv_msg.data(), recv_msg.size());
        if (resp.type() == fasterdp::Response::INSTRUCTION && resp.instruction().valid()) {
            auto &inst = resp.instruction();
            std::cerr << "Instruction: iter=" << inst.effective_since_uiter()
                 << ", bs=" << inst.config().ubatch_size() 
                << ", ga=" << inst.config().gradient_accumulation()
                << ", cpr=" << inst.config().compression_rate() << std::endl;

            pair.second(inst.effective_since_uiter(), TrainConfig({
                true,
                inst.config().ubatch_size(),
                inst.config().gradient_accumulation(),
                inst.config().compression_rate()
            }));
        }
    }
}
