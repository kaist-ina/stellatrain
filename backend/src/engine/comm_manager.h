#ifndef ENGINE_COMM_MANAGER_H
#define ENGINE_COMM_MANAGER_H
#include <cstdint>
#include <string>
#include <thread>
#include <memory>
#include <queue>
#include <zmq.hpp>
#include <map>
#include <set>
#include <condition_variable>
#include <mutex>
#include <array>
#include <functional>
#include <future>
#include <unordered_map>
#include <memory>
#include "message.h"
#include "../misc/bandwidth_monitor.h"

class FasterDpEngine;
class TrainTaskV2;

const uint8_t COMM_FLAG_UINT16_IDX = 0x01;
const uint8_t COMM_FLAG_FP16_VAL = 0x02;

typedef uint16_t fp16_t;

enum CommControlState {
    COMM_WAIT_INIT,
    COMM_READY
};

struct zmqid_t {
    char identity[5];
    MsgInitReq req;
    int64_t timestamp_offset;
};

struct SeqEnvelope {
    bool valid;
    zmqid_t identity;
    zmq::message_t msg;
};

struct RecvBufSpec {
    bool is_malloc_segment;
    float *val;
    uint32_t *idx;
    size_t len;
    int64_t sender_ts;
    int64_t receiver_ts;
};

struct TxChunkSpec {
    int priority;
    uint8_t flag;
    std::string key;
    int64_t numel;
    uint32_t *ptr_grad_idx;
    uint16_t *ptr_grad_idx_16;
    float *ptr_grad_val;
    fp16_t *ptr_grad_val_16;
    unsigned iter;
};

class CommManager {
    FasterDpEngine *engine_;
    zmq::context_t zmq_ctx_;
    
    zmq::socket_t sock_router_;

    zmq::socket_t sock_ctrl_;
    const std::string addr_;
    const uint16_t port_;
    CommControlState state_;
    std::map<int, zmqid_t> client_zmq_id_map_;
    std::map<uint16_t, uint16_t> receier_id_to_sender_id_map_;
    
    int client_id_;
    int num_clients_;
    int num_active_clients_;

    bool finished_;
    
    BandwidthMonitor bandwidth_monitor_;

    void server_thread_main();
    SeqEnvelope recv_req_envelope(zmq::socket_t &sock, zmq::recv_flags flags = zmq::recv_flags::none);
    std::unique_ptr<std::thread> server_thread_;

    mutable std::mutex wait_client_connect_mutex_;
    mutable std::condition_variable wait_client_connect_cond_;

    void pull_thread_main();
    std::unique_ptr<std::thread> pull_thread_;

    void push_thread_main();
    std::unique_ptr<std::thread> push_thread_;

    struct Compare {
        bool operator()(const TxChunkSpec& a, const TxChunkSpec& b) {
            return a.priority > b.priority;
        }
    };

    std::priority_queue<TxChunkSpec, std::vector<TxChunkSpec>, Compare> tx_queue_;
    std::mutex tx_queue_mutex_;
    std::condition_variable tx_queue_cond_;
    
    void debug_thread_main();
    std::unique_ptr<std::thread> debug_thread_;
    std::map<std::string, std::pair<std::shared_ptr<std::packaged_task<void ()>>, RecvBufSpec>> pull_callback_map_;
    std::map<std::string, const TrainTaskV2 *> sent_log_map_;
    mutable std::mutex pull_callback_map_mutex_;

    std::unordered_map<std::string, std::pair<int, std::unique_ptr<uint8_t []>>> init_model_proxy_map_;
    std::unordered_map<std::string, std::vector<zmqid_t>> init_model_pending_vec_map_;
    std::unordered_map<std::string, std::size_t> num_clients_waiting_for_model_map_; 



    std::unordered_map<std::string, std::pair<float *, uint32_t *>> delegate_delete_map_;
    std::unordered_map<std::string, int> delegate_sent_cnt_map_;
    std::mutex delegate_delete_mutex_;

    uint16_t find_client_id_by_zmqid(const zmqid_t &zmqid) const;
public:

    zmq::socket_t sock_pull_;
    zmq::socket_t sock_push_;
    std::mutex sock_mutex_;

    CommManager(const std::string addr, uint16_t port, FasterDpEngine *engine = nullptr);
    ~CommManager();

    void startServer(int num_clients);
    void startClient(int client_id);

    void waitClientConnect() const;

    void queueTx(const TrainTaskV2 *task, const uint32_t *ptr_grad_idx, const float *ptr_grad_val, unsigned iter);

    void sendInitmodel(const std::string skey, const float *ptr_model_val, const uint32_t len);
    RecvModelSpec recvInitmodel(std::string skey);

    void sendStatReport(int64_t sender_timestamp, int64_t receiver_timestamp, uint32_t payload_size);
    void initiateTimeSync();
    MsgConfigUpdateResponse reportProbeLoss(int batch_size, double loss, double compression_ratio);

    inline std::mutex & get_pull_callback_map_mutex() const {
        return pull_callback_map_mutex_;
    }

    inline std::map<std::string, std::pair<std::shared_ptr<std::packaged_task<void ()>>, RecvBufSpec>> & get_pull_callback_map() {
        return pull_callback_map_;
    }

    void stat_debug_print() const;

    void delegate_delete(const TrainTaskV2 *task);
};


#endif