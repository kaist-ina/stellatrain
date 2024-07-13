
#ifndef _ENGINE_MESSAGE_H_
#define _ENGINE_MESSAGE_H_

#include <cstdint>
#include <string>
#include <memory>
#include <netinet/in.h>

struct MsgInitReq {
    uint16_t client_id;
    char host_alias[64];
    in_addr_t public_ip;
    uint16_t listen_port;
    int64_t timestamp;
    uint16_t num_gpus;
};


struct MsgInitRes {
    bool success;
    char target_host_alias[64];
    in_addr_t target_ip;
    uint16_t target_port;
};

struct RecvModelSpec {
    std::string skey;
    std::unique_ptr<float []> ptr;
    size_t len;
};

enum class BrokerMsgType : uint8_t {
    MODEL_REP = 0,
    STAT_REP,
    SYNC_TIME,
    CONFIG_UPD,
    LOSS_PROBE,
    TELEMETRY_INIT,
    TELEMETRY_INIT_RESP,
    TELEMETRY_REPORT,
};

struct MsgStatReport {
    int64_t sender_timestamp;
    int64_t receiver_timestamp;
    uint32_t payload_size;
};


struct MsgSyncTime {
    int64_t timestamp;
};


struct MsgConfigUpdateReport {
    double bandwidth;
    int batch_size;
    double compression_ratio;
    double loss;
};


struct MsgConfigUpdateResponse {
    double compression_ratio;
    int batch_size;
    int iter;

    int probe_batch_size[2];
    double probe_compression_ratio[2];
};


struct MsgTelemetrySessionInit {
    uint16_t rank;
    uint16_t local_rank;
    uint16_t local_world_size;
    char gpu_name[64];
};

struct MsgTelemetrySessionInitResp {
    uint16_t client_id;
};


struct TrainConfig {
    bool valid;
    uint32_t ubatch_size;
    uint32_t gradient_accumulation;
    float compression_rate;

    TrainConfig() = default;
    TrainConfig(bool valid, uint32_t ubatch_size, uint32_t gradient_accumulation, float compression_rate) :
        valid(valid), ubatch_size(ubatch_size), gradient_accumulation(gradient_accumulation), compression_rate(compression_rate) {}
};

struct TrainMeasurement {
    float grad_send_thpt_mbps;
    float grad_recv_thpt_mbps;
    float fwd_delay_ms;
    float bwd_delay_ms;
    float bwd_fwd_gap_ms;
    float intra_fwd_gap_ms;

    TrainMeasurement() = default;
    TrainMeasurement(float grad_send_thpt_mbps, float grad_recv_thpt_mbps, float fwd_delay_ms, float bwd_delay_ms, float bwd_fwd_gap_ms, float intra_fwd_gap_ms) :
        grad_send_thpt_mbps(grad_send_thpt_mbps), grad_recv_thpt_mbps(grad_recv_thpt_mbps), fwd_delay_ms(fwd_delay_ms), bwd_delay_ms(bwd_delay_ms), bwd_fwd_gap_ms(bwd_fwd_gap_ms), intra_fwd_gap_ms(intra_fwd_gap_ms) {}
};


struct MsgTelemetry {
    uint16_t telemetry_client_id;
    uint32_t iter; // iter * grad_acc + uiter
    TrainConfig config;
    TrainMeasurement measurement;
};

struct MsgInstruction {
    uint32_t iter_effective_since;
    TrainConfig config;
};
#endif