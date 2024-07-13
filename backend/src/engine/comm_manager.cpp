
#include "comm_manager.h"
#include "core.h"
#include "message.h"
#include "../misc/ipaddr.h"
#include <iostream>
#include <string>
#include <array>
#include <exception>
#include <zmq_addon.hpp>
#include <random>
#include <vector>
#include <stdexcept>
#include <pthread.h>
#include <csignal>
#include <time.h>

static void hexDump(const void* data, size_t size) {
    const unsigned char* p = (const unsigned char*)data;

    for (size_t i = 0; i < size; ++i) {
        printf("%02X ", p[i]);

        // Print an extra space after every 8 bytes
        if ((i + 1) % 8 == 0)
            printf(" ");

        // Print an extra newline after every 16 bytes
        if ((i + 1) % 16 == 0)
            printf("\n");
    }

    // Add a newline if the last line was not complete
    if (size % 16 != 0)
        printf("\n");
}

inline static int64_t get_timestamp() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000000000 + ts.tv_nsec;
}

static std::string hexDumpStr(const void* data, size_t size) {
    const unsigned char* p = (const unsigned char*)data;
    auto buf = std::make_unique<char []>(size * 3);
    memset(buf.get(), 0, size * 3);
    char *pos = buf.get();

    for (size_t i = 0; i < size; ++i) {
        pos += sprintf(pos, "%02X ", p[i]);
    }

    return std::string(buf.get());
}

static uint16_t chooseListenPort() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(10240, 49151);

    return static_cast<uint16_t>(dis(gen));
}


CommManager::CommManager(const std::string addr, uint16_t port, FasterDpEngine *engine): 
    addr_(addr), port_(port), state_(COMM_WAIT_INIT), client_id_(-1), num_clients_(-1), finished_(false), num_active_clients_(0), engine_(engine) {
}


void CommManager::startServer(int num_clients) {
    num_clients_ = num_clients;
    // setup server
    server_thread_ = std::make_unique<std::thread>(&CommManager::server_thread_main, this);
}


SeqEnvelope CommManager::recv_req_envelope(zmq::socket_t &sock, zmq::recv_flags flags) {
    SeqEnvelope env;
    env.valid = false;

    std::array<zmq::message_t, 3> recv_msgs; // possibility of double copy
    
    auto result = zmq::recv_multipart_n(sock, recv_msgs.data(), 3);
    if (!result.has_value())
        return env;
        
    memcpy(&env.identity, recv_msgs[0].data(), sizeof(env.identity));
    env.msg.move(recv_msgs[2]);
    env.valid = true;
    // std::cerr << "Message from " << hexDumpStr(&env.identity, sizeof(env.identity)) << " : " << env.msg.to_string() << std::endl;
    return env;
}

void CommManager::server_thread_main() {
    sigset_t mask;
    sigemptyset (&mask);
    sigaddset (&mask, SIGTERM);
    sigaddset (&mask, SIGHUP);
    pthread_sigmask(SIG_BLOCK, &mask, NULL);
    pthread_setname_np(pthread_self(), "CommBroker");

    sock_router_ = zmq::socket_t(zmq_ctx_, zmq::socket_type::router);
    std::cerr << "Try bind..." << std::endl;
    const std::string bind_addr = std::string("tcp://*:") + std::to_string(port_);
    sock_router_.bind(bind_addr);
    std::cerr << "Bind OK" << std::endl;
    zmqid_t master_identity;
    
    std::vector<int> client_ids;

    while (num_active_clients_ != num_clients_) {
        auto env = recv_req_envelope(sock_router_);
        auto req = reinterpret_cast<MsgInitReq *>(env.msg.data());
        int64_t timestamp_recv = get_timestamp();
        int64_t timestamp_offset = req->timestamp - timestamp_recv;
        client_zmq_id_map_[req->client_id] = env.identity;
        client_zmq_id_map_[req->client_id].req = *req;
        client_zmq_id_map_[req->client_id].timestamp_offset = timestamp_offset;
        client_ids.push_back(req->client_id);
        std::cerr << "Connected client " << req->client_id << "(" << num_active_clients_ << " / " << num_clients_ << "), timestamp_offset = "  << (timestamp_offset / 1000) << " us" << std::endl;

        num_active_clients_++;

        if (req->client_id == 0) {
            master_identity = env.identity;
        }

    }

    MsgInitRes res;
    res.success = true;

    for(unsigned i = 0; i < client_ids.size(); i++) {

        // naive algorithm here
        const auto current_client_id = client_ids[i];
        const auto target_client_id = client_ids[(i + 1) % client_ids.size()];
        const auto &current_client = client_zmq_id_map_[current_client_id];
        const auto &target_client = client_zmq_id_map_[target_client_id];

        receier_id_to_sender_id_map_[target_client_id] = current_client_id;
        std::cerr << "Client " << current_client_id << " (" << current_client.req.host_alias << ")"  << " -> " << target_client_id << " (" << target_client.req.host_alias << ")" << std::endl;

        strncpy(res.target_host_alias, target_client.req.host_alias, sizeof(target_client.req.host_alias));
        res.target_ip = target_client.req.public_ip;
        res.target_port = target_client.req.listen_port;

        std::array<zmq::const_buffer, 3> send_msgs = {
            zmq::buffer(current_client.identity, sizeof(current_client.identity)),
            zmq::buffer(current_client.identity, 0),
            zmq::buffer(&res, sizeof(MsgInitRes)),
        };

        zmq::send_multipart(sock_router_, send_msgs);
    }
    wait_client_connect_cond_.notify_all();

    const auto num_clients = client_ids.size();

    while (!finished_) {
        auto env = recv_req_envelope(sock_router_);

        /**
         * | type | len | key | payload... |
        */

        if (static_cast<BrokerMsgType>(env.msg.data<char>()[0]) == BrokerMsgType::MODEL_REP) {

            bool from_master = false;
            if (memcmp(master_identity.identity, env.identity.identity, sizeof(master_identity.identity)) == 0) {            
                from_master = true;
                std::string key(env.msg.data<char>() + 2, (int)(env.msg.data<char>()[1]));

                std::cerr << "Received from master : key=" << std::string(env.msg.data<char>() + 2, (int)(env.msg.data<char>()[1])) << std::endl;

                std::size_t remaining_clients = num_clients - 1;

                // if previously got request, send to that request
                if (init_model_pending_vec_map_.find(key) != init_model_pending_vec_map_.end()) {
                    auto t = init_model_pending_vec_map_.find(key);
                    for (auto it = t->second.begin(); it != t->second.end(); ++it) {
                        assert(remaining_clients > 0);
                        std::array<zmq::const_buffer, 3> send_msgs = {
                            zmq::buffer(it->identity, sizeof(env.identity)),
                            zmq::buffer(it->identity, 0),
                            zmq::buffer(env.msg.data<char>(), env.msg.size()),
                        };
                        zmq::send_multipart(sock_router_, send_msgs);
                        // std::cerr << "Immediately serving cached requests for " << key  << std::endl;
                        remaining_clients--;
                    }
                    init_model_pending_vec_map_.erase(key);
                }

                if (remaining_clients > 0) {
                    // save to init_model_proxy_map_ and num_clients_waiting_for_model_map_
                    assert(num_clients_waiting_for_model_map_.find(key) == num_clients_waiting_for_model_map_.end());
                    num_clients_waiting_for_model_map_[key] = remaining_clients;

                    assert(init_model_proxy_map_.find(key) == init_model_proxy_map_.end());
                    auto buf = std::make_unique<uint8_t []>(env.msg.size());
                    memcpy(buf.get(), env.msg.data<uint8_t>(), env.msg.size());

                    init_model_proxy_map_.insert(std::make_pair(key, std::make_pair(env.msg.size(), std::move(buf))));
                    std::cerr << "Stashing data for " << key  << std::endl;
                } else {
                    if (num_clients_waiting_for_model_map_.find(key) != num_clients_waiting_for_model_map_.end()) {
                        num_clients_waiting_for_model_map_.erase(key);
                    }
                }

                // send dummy for sender

                std::array<zmq::const_buffer, 3> send_msgs = {
                    zmq::buffer(env.identity.identity, sizeof(env.identity)),
                    zmq::buffer(env.identity.identity, 0),
                    zmq::str_buffer("")
                };
                zmq::send_multipart(sock_router_, send_msgs);
            } else {
                // std::cerr << "Received from secondary" << std::endl;
                std::string key(env.msg.data<char>() + 2, env.msg.size() - 2);
                if (init_model_proxy_map_.find(key) == init_model_proxy_map_.end()) {
                    
                    // std::cerr << "Stashing request for " << key  << std::endl;
                    init_model_pending_vec_map_[key].push_back(env.identity);
                } else {
                    // std::cerr << "Immediately serving request for " << key  << std::endl;
                    auto &pair = init_model_proxy_map_[key];

                    std::array<zmq::const_buffer, 3> send_msgs = {
                        zmq::buffer(env.identity.identity, sizeof(env.identity)),
                        zmq::buffer(env.identity.identity, 0),
                        zmq::buffer(pair.second.get(), pair.first),
                    };

                    zmq::send_multipart(sock_router_, send_msgs);
                    
                    assert(num_clients_waiting_for_model_map_.find(key) != num_clients_waiting_for_model_map_.end());
                    num_clients_waiting_for_model_map_[key]--;
                    if (num_clients_waiting_for_model_map_[key] == 0) {
                        num_clients_waiting_for_model_map_.erase(key);
                        init_model_proxy_map_.erase(key);
                        // std::cerr << "Removing cached elements for " << key  << std::endl;
                    }

                    // std::cerr << "Sent to secondary" << std::endl;
                }
            }

        } else if (static_cast<BrokerMsgType>(env.msg.data<char>()[0]) == BrokerMsgType::STAT_REP) {
            auto *msg = reinterpret_cast<MsgStatReport *>(env.msg.data<char>() + 1);

            int32_t client_id = -1;
            for (auto it = client_zmq_id_map_.begin(); it != client_zmq_id_map_.end(); ++it) {
                if (memcmp(it->second.identity, env.identity.identity, sizeof(it->second.identity)) == 0) {
                    client_id = it->first;
                    break;
                }
            }
            assert(client_id >= 0);
            int32_t sender_id = receier_id_to_sender_id_map_[client_id];

            // if (memcmp(master_identity.identity, env.identity.identity, sizeof(master_identity.identity)) == 0) { 
            auto start = msg->sender_timestamp - client_zmq_id_map_[sender_id].timestamp_offset;
            auto end = msg->receiver_timestamp - client_zmq_id_map_[client_id].timestamp_offset;
            auto duration_ms = (double)(end-start) * 1e-6;
            auto thpt_mbps = (double)msg->payload_size * 8 / duration_ms / 1000.;
            
            
            std::cerr << "Received stat report from " << sender_id << " -> " << client_id 
                // << ", ts=" << msg->sender_timestamp << " -> " << msg->receiver_timestamp 
                << ", ts_diff_ms=" << duration_ms << ", bytes = " << msg->payload_size << ", thpt_mbps=" << thpt_mbps << std::endl;
            
            msg->receiver_timestamp = get_timestamp();
            msg->payload_size = env.msg.size();

            std::array<zmq::const_buffer, 3> send_msgs = {
                zmq::buffer(env.identity.identity, sizeof(env.identity)),
                zmq::buffer(env.identity.identity, 0),
                zmq::str_buffer(""),
            };
            zmq::send_multipart(sock_router_, send_msgs);
            
        } else if (static_cast<BrokerMsgType>(env.msg.data<char>()[0]) == BrokerMsgType::SYNC_TIME) {
            auto *msg = reinterpret_cast<MsgSyncTime *>(env.msg.data<char>() + 1);
            int64_t timestamp_recv = get_timestamp();
            int64_t timestamp_offset = msg->timestamp - timestamp_recv;

            int32_t client_id = -1;
            for (auto it = client_zmq_id_map_.begin(); it != client_zmq_id_map_.end(); ++it) {
                if (memcmp(it->second.identity, env.identity.identity, sizeof(it->second.identity)) == 0) {
                    client_id = it->first;
                    break;
                }
            }
            assert(client_id >= 0);

            auto prev_offset = client_zmq_id_map_[client_id].timestamp_offset;
            auto new_offset = (prev_offset * 0.9 + timestamp_offset * 0.1);
            auto offset_diff = new_offset - prev_offset;
            client_zmq_id_map_[client_id].timestamp_offset = new_offset;
            std::cerr << "Received sync time from " << client_id << ", offset = " << ((new_offset - prev_offset) / 1000) << " us" << std::endl;
            
            std::array<zmq::const_buffer, 3> send_msgs = {
                zmq::buffer(env.identity.identity, sizeof(env.identity)),
                zmq::buffer(env.identity.identity, 0),
                zmq::str_buffer(""),
            };
            zmq::send_multipart(sock_router_, send_msgs);
        } else if (static_cast<BrokerMsgType>(env.msg.data<char>()[0]) == BrokerMsgType::CONFIG_UPD) {
            auto *msg = reinterpret_cast<MsgConfigUpdateReport *>(env.msg.data<char>() + 1);
            int32_t client_id = find_client_id_by_zmqid(env.identity);
            
            // std::cerr << "Received bw update from " << client_id << ", bandwidth = " << msg->bandwidth << " Mbps" << std::endl;

            char res_buf[sizeof(MsgConfigUpdateResponse)+1] = {0, };
            auto *res = reinterpret_cast<MsgConfigUpdateResponse *>(res_buf + 1);
            
            // need to set something here..
            
            std::array<zmq::const_buffer, 3> send_msgs = {
                zmq::buffer(env.identity.identity, sizeof(env.identity)),
                zmq::buffer(env.identity.identity, 0),
                zmq::buffer(res_buf, sizeof(MsgConfigUpdateResponse) + 1),
            };
            zmq::send_multipart(sock_router_, send_msgs);

        } else if (static_cast<BrokerMsgType>(env.msg.data<char>()[0]) == BrokerMsgType::LOSS_PROBE) {

        }
    }

}


uint16_t CommManager::find_client_id_by_zmqid(const zmqid_t &zmqid) const {
    for (auto it = client_zmq_id_map_.begin(); it != client_zmq_id_map_.end(); ++it) {
        if (memcmp(it->second.identity, &zmqid, sizeof(it->second.identity)) == 0) {
            return it->first;
        }
    }
    assert(0);
    return 0;
}


CommManager::~CommManager() {
    finished_ = true;
    zmq_ctx_.shutdown();

    sock_push_.close();
    sock_pull_.close();

    if (server_thread_) {
        server_thread_->join();
    }
    if (pull_thread_) {
        pull_thread_->join();
    }
    if (push_thread_) {
        tx_queue_cond_.notify_one();
        push_thread_->join();
    }
}


void CommManager::startClient(int client_id) {

    client_id_ = client_id;


    const auto my_hostname = get_host_name();
    const auto my_canon_name = get_my_canonical_name();
    const auto my_public_ip = get_my_public_ip();

    std::cerr << "Host info: " << my_hostname << " / " << my_canon_name << " / " << addr_ntos(my_public_ip) << std::endl;
    std::cerr << "Creating sockets..." << std::endl;
    sock_ctrl_ = zmq::socket_t(zmq_ctx_, zmq::socket_type::req);
    sock_pull_ = zmq::socket_t(zmq_ctx_, zmq::socket_type::pull);
    sock_push_ = zmq::socket_t(zmq_ctx_, zmq::socket_type::push);

    int sock_val = 0;
    size_t sock_val_len = 4;
    
    if (zmq_setsockopt(sock_pull_, ZMQ_SNDHWM, &sock_val, sock_val_len) != 0) {
        perror("zmq_getsockopt");
    }
    if (zmq_setsockopt(sock_pull_, ZMQ_RCVHWM, &sock_val, sock_val_len) != 0) {
        perror("zmq_getsockopt");
    }
    if (zmq_setsockopt(sock_push_, ZMQ_SNDHWM, &sock_val, sock_val_len) != 0) {
        perror("zmq_getsockopt");
    }
    if (zmq_setsockopt(sock_push_, ZMQ_RCVHWM, &sock_val, sock_val_len) != 0) {
        perror("zmq_getsockopt");
    }
    

    if (zmq_getsockopt(sock_pull_, ZMQ_SNDHWM, &sock_val, &sock_val_len) != 0) {
        perror("zmq_getsockopt");
    }
    std::cout << "Sock Pull SNDHWM = " << sock_val << std::endl;
    assert(sock_val == 0);

    if (zmq_getsockopt(sock_pull_, ZMQ_RCVHWM, &sock_val, &sock_val_len) != 0) {
        perror("zmq_getsockopt");
    }
    std::cout << "Sock Pull RCVHWM = " << sock_val << std::endl;
    assert(sock_val == 0);

    if (zmq_getsockopt(sock_push_, ZMQ_SNDHWM, &sock_val, &sock_val_len) != 0) {
        perror("zmq_getsockopt");
    }
    std::cout << "Sock Push SNDHWM = " << sock_val << std::endl;
    assert(sock_val == 0);

    if (zmq_getsockopt(sock_push_, ZMQ_RCVHWM, &sock_val, &sock_val_len) != 0) {
        perror("zmq_getsockopt");
    }
    std::cout << "Sock Push RCVHWM = " << sock_val << std::endl;
    assert(sock_val == 0);


    uint16_t my_listen_port;

    while (true) {
        my_listen_port = chooseListenPort();
        try {
            sock_pull_.bind(std::string("tcp://*:" + std::to_string(my_listen_port)));
        } catch (const std::runtime_error& e) {
            continue;
        }
        break;
    }
    std::cerr << "My Pull Socket is bound at " << addr_ntos(my_public_ip) << ":" << my_listen_port << std::endl;


    // try connecting server and block
    std::cerr << "Connneting to broker... ";
    sock_ctrl_.connect(std::string("tcp://") + addr_ + ":" + std::to_string(port_));

    std::cerr << "Connect OK. " << std::endl;
    std::cerr << "Retrieving configuration... ";
    MsgInitReq req;
    req.client_id = client_id;
    req.public_ip = my_public_ip;
    req.listen_port = my_listen_port;
    req.timestamp = get_timestamp();
    strncpy(req.host_alias, my_hostname.c_str(), sizeof(req.host_alias));

    sock_ctrl_.send(zmq::buffer(&req, sizeof(MsgInitReq)));
    std::cerr << "Sending Req... ";

    zmq::message_t msg_recv;
    auto result = sock_ctrl_.recv(msg_recv);
    if (!result.has_value()) {
        throw std::runtime_error("Failed to initialize connection.");
    }
    MsgInitRes *res = reinterpret_cast<MsgInitRes *>(msg_recv.data());
    if (!res->success) {
        throw std::runtime_error("Failed to initialize connection.");
    }

    std::cerr << "Config OK!" << std::endl;
    const std::string target_uri =  std::string("tcp://") + addr_ntos(res->target_ip) + ":" + std::to_string(res->target_port);
    std::cerr << "Push to " << res->target_host_alias << "(" << target_uri << ")" << std::endl;
    std::cerr << "Connecting Push-Pull sockets..." << std::endl;
    sock_push_.connect(target_uri);

    std::cerr << "Connect OK!" << std::endl;

    pull_thread_ = std::make_unique<std::thread>(&CommManager::pull_thread_main, this);
    push_thread_ = std::make_unique<std::thread>(&CommManager::push_thread_main, this);
    // debug_thread_ = std::make_unique<std::thread>(&CommManager::debug_thread_main, this);
}

void CommManager::waitClientConnect() const {
    std::unique_lock<std::mutex> ul(wait_client_connect_mutex_);
    while (num_active_clients_ < num_clients_ && !finished_) {
        wait_client_connect_cond_.wait(ul);
    }
}


static void cast_uint16_to_uint32(const uint16_t* src, uint32_t* dst, size_t length) {
    size_t i = 0;

    // Process 8 uint16_t elements at a time
    for (; i + 8 < length; i += 8) {
        // Load 8 uint16_t values
        __m128i vec = _mm_loadu_si128((const __m128i*)(src + i));

        // Extend to 8 uint32_t values using _mm256_cvtepi16_epi32
        __m256i result = _mm256_cvtepi16_epi32(vec);

        // Store the result to the destination
        _mm256_storeu_si256((__m256i*)(dst + i), result);
    }

    // Handle any remaining elements
    for (; i < length; i++) {
        dst[i] = src[i];
    }
}


static void cast_uint32_to_uint16(const uint32_t* src, uint16_t* dst, size_t length) {
    size_t i = 0;

    // Process 8 uint32_t elements at a time
    for (; i + 8 < length; i += 8) {
        // Load 8 uint32_t values
        __m128i vecA = _mm_loadu_si128((const __m128i*)(src + i));
        __m128i vecB = _mm_loadu_si128((const __m128i*)(src + i + 4));

        // Use _mm_packs_epi32 to pack/saturate cast 32-bit integers to 16-bit integers
        __m128i result = _mm_packs_epi32(vecA, vecB);

        // Store the result to the destination
        _mm_storeu_si128((__m128i*)(dst + i), result);
    }

    // Handle any remaining elements
    for (; i < length; i++) {
        dst[i] = (uint16_t)src[i];
    }
}

static void cast_fp32_to_fp16(const float* src, fp16_t* dst, size_t length) {
    size_t i = 0;

    // Process 8 float elements at a time
    for (; i + 8 < length; i += 8) {
        // Load 8 float values
        __m256 vec = _mm256_loadu_ps(src + i);

        // Cast to 8 half-precision float values using _mm256_cvtps_ph
        __m128i result = _mm256_cvtps_ph(vec, 0);

        // Store the result to the destination
        _mm_storeu_si128((__m128i*)(dst + i), result);
    }

    // Handle any remaining elements
    for (; i < length; i++) {
        dst[i] = src[i];
    }
}

static void cast_fp16_to_fp32(const fp16_t* src, float* dst, size_t length) {
    size_t i = 0;

    // Process 8 __fp16 elements at a time
    for (; i + 8 < length; i += 8) {
        // Load 8 __fp16 values
        __m128i vec = _mm_loadu_si128((const __m128i*)(src + i));

        // Cast to 8 float values using _mm256_cvtph_ps
        __m256 result = _mm256_cvtph_ps(vec);

        // Store the result to the destination
        _mm256_storeu_ps(dst + i, result);
    }

    // Handle any remaining elements
    for (; i < length; i++) {
        dst[i] = src[i];
    }
}
#if PRIORITY_TX
void CommManager::queueTx(const TrainTaskV2 *task, const uint32_t *ptr_grad_idx, const float *ptr_grad_val, unsigned iter) {
    uint16_t *ptr_grad_idx_16 = nullptr;
    fp16_t *ptr_grad_val_16 = nullptr;
    uint8_t flag = 0;

#if IDX_COMPRESSION
    if (task->tensor_numel_ < 65536) {
        ptr_grad_idx_16 = new uint16_t[task->tensor_compressed_numel_];
        cast_uint32_to_uint16(ptr_grad_idx, ptr_grad_idx_16, task->tensor_compressed_numel_);
        flag |= COMM_FLAG_UINT16_IDX;
    }
#endif

#if FP16_COMPRESSION
    ptr_grad_val_16 = new fp16_t[task->tensor_compressed_numel_];
    cast_fp32_to_fp16(ptr_grad_val, ptr_grad_val_16, task->tensor_compressed_numel_);
    flag |= COMM_FLAG_FP16_VAL;
#endif

    std::unique_lock<std::mutex> ul(tx_queue_mutex_);
    tx_queue_.push({task->priority(), 
        flag,
        std::string(task->key()),
        task->tensor_compressed_numel_,
        const_cast<uint32_t *>(ptr_grad_idx), 
        ptr_grad_idx_16,
        const_cast<float *>(ptr_grad_val), 
        ptr_grad_val_16, iter});
    tx_queue_cond_.notify_one();
}
#else
void CommManager::queueTx(const TrainTaskV2 *task, const uint32_t *ptr_grad_idx, const float *ptr_grad_val, unsigned iter) {
    uint16_t *ptr_grad_idx_16 = nullptr;
    fp16_t *ptr_grad_val_16 = nullptr;
    uint8_t flag = 0;

#if IDX_COMPRESSION
    if (task->tensor_numel_ < 65536) {
        ptr_grad_idx_16 = new uint16_t[task->tensor_compressed_numel_];
        cast_uint32_to_uint16(ptr_grad_idx, ptr_grad_idx_16, task->tensor_compressed_numel_);
        flag |= COMM_FLAG_UINT16_IDX;
    }
#endif

#if FP16_COMPRESSION
    ptr_grad_val_16 = new fp16_t[task->tensor_compressed_numel_];
    cast_fp32_to_fp16(ptr_grad_val, ptr_grad_val_16, task->tensor_compressed_numel_);
    flag |= COMM_FLAG_FP16_VAL;
#endif

    std::unique_lock<std::mutex> ul(sock_mutex_);
    const std::string key(task->key() + "!" + std::to_string(iter));
    // sent_log_map_[key] = task;
    std::array<zmq::const_buffer, 4> send_msgs = {
        zmq::buffer(key.c_str(), key.length()),
        zmq::buffer(&flag, sizeof(flag)),
        (flag & COMM_FLAG_UINT16_IDX) ?
        zmq::buffer(ptr_grad_idx_16, task-> tensor_compressed_numel_ * sizeof(uint16_t)):
        zmq::buffer(ptr_grad_idx, task-> tensor_compressed_numel_ * sizeof(uint32_t)),
        
        (flag & COMM_FLAG_FP16_VAL) ?
        zmq::buffer(ptr_grad_val_16, task-> tensor_compressed_numel_ * sizeof(fp16_t)) :
        zmq::buffer(ptr_grad_val, task-> tensor_compressed_numel_ * sizeof(float))
    };
    // std::cerr << "tx queued " << task->key() << std::endl;
    
    zmq::send_multipart(sock_push_, send_msgs, zmq::send_flags::dontwait);
}
#endif

void CommManager::stat_debug_print() const {
    std::cerr << "===sent_log===" << std::endl;
    for(auto it = sent_log_map_.begin(); it != sent_log_map_.end(); ++it) {
        std::cerr << " > " <<  it->first << " : " << std::ios::hex << reinterpret_cast<uintptr_t>(it->second) << std::ios::dec << std::endl;
    }
    std::cerr << "====================" << std::endl;

    std::cerr << "===pull_callback_map_===" << std::endl;
    for(auto it = pull_callback_map_.begin(); it != pull_callback_map_.end(); ++it) {
        std::cerr << " < " <<  it->first << " : " << it->second.first << std::endl;
    }
    std::cerr << "====================" << std::endl;

}


MsgConfigUpdateResponse CommManager::reportProbeLoss(int batch_size, double loss, double compression_ratio) {
    char buf[sizeof(MsgConfigUpdateReport) + 1] = {0, };
    MsgConfigUpdateReport *msg = reinterpret_cast<MsgConfigUpdateReport *>(buf + 1);
    buf[0] = static_cast<char>(BrokerMsgType::LOSS_PROBE);

    msg->batch_size = batch_size;
    msg->compression_ratio = 0;
    msg->loss = loss;
    
    {
        std::unique_lock<std::mutex> ul(sock_mutex_);
        sock_ctrl_.send(zmq::buffer(buf, sizeof(MsgStatReport) + 1));
    }

    {
        std::unique_lock<std::mutex> ul(sock_mutex_);
        zmq::message_t recv_msg;
        auto result = sock_ctrl_.recv(recv_msg);
        auto *msg = reinterpret_cast<MsgConfigUpdateResponse *>(recv_msg.data<char>() + 1);
        MsgConfigUpdateResponse resp;
        memcpy(&resp, msg, sizeof(MsgConfigUpdateResponse));
        return resp;
    } 

}


void CommManager::debug_thread_main() {

    sigset_t mask;
    sigemptyset (&mask);
    sigaddset (&mask, SIGTERM);
    sigaddset (&mask, SIGHUP);
    pthread_sigmask(SIG_BLOCK, &mask, NULL);
    pthread_setname_np(pthread_self(), "CommDbgThread");


    while (!finished_) {
        // stat_debug_print();
        auto mbps = bandwidth_monitor_.query_bandwidth_mbps();
        if (mbps > 0) {
            // report and update bandwidth
            char buf[sizeof(MsgConfigUpdateReport) + 1] = {0, };
            MsgConfigUpdateReport *msg = reinterpret_cast<MsgConfigUpdateReport *>(buf + 1);
            buf[0] = static_cast<char>(BrokerMsgType::CONFIG_UPD);
            msg->bandwidth = mbps;
            {
                std::unique_lock<std::mutex> ul(sock_mutex_);
                sock_ctrl_.send(zmq::buffer(buf, sizeof(MsgStatReport) + 1));
            }

            // recv dummy message here
            {
                std::unique_lock<std::mutex> ul(sock_mutex_);
                zmq::message_t recv_msg;
                auto result = sock_ctrl_.recv(recv_msg);
            } 
        }
        usleep(250000);
    }
}


void CommManager::push_thread_main() {
    sigset_t mask;
    sigemptyset (&mask);
    sigaddset (&mask, SIGTERM);
    sigaddset (&mask, SIGHUP);
    pthread_sigmask(SIG_BLOCK, &mask, NULL);
    pthread_setname_np(pthread_self(), "CommPushThread");

    while (!finished_) {
        TxChunkSpec spec;
        {
            std::unique_lock<std::mutex> ul(tx_queue_mutex_);
            tx_queue_cond_.wait(ul, [this] { return !tx_queue_.empty() || finished_; });
            if (finished_) {
                break;
            }
            if (tx_queue_.empty()) {
                continue;
            }

            spec = tx_queue_.top();
            tx_queue_.pop();
        }

        {
            std::unique_lock<std::mutex> ul(sock_mutex_);
            const std::string key(spec.key + "!" + std::to_string(spec.iter));

            // print key, ptr_grad_idx, val
            // std::cerr << "Sending " << key << " (" << spec.priority << ")" << std::endl;
            auto sender_timestamp = get_timestamp();
            std::array<zmq::const_buffer, 4> send_msgs = {
                zmq::buffer(key.c_str(), key.length()),
                zmq::buffer(&spec.flag, sizeof(spec.flag)),
                (spec.flag & COMM_FLAG_UINT16_IDX) ?
                    zmq::buffer(spec.ptr_grad_idx_16, spec.numel * sizeof(uint16_t)) :
                    zmq::buffer(spec.ptr_grad_idx, spec.numel * sizeof(uint32_t)),

                (spec.flag & COMM_FLAG_FP16_VAL) ?
                    zmq::buffer(spec.ptr_grad_val_16, spec.numel * sizeof(fp16_t)) : 
                    zmq::buffer(spec.ptr_grad_val, spec.numel * sizeof(float))
            };
            zmq::send_multipart(sock_push_, send_msgs, zmq::send_flags::dontwait);

        }

#if IDX_COMPRESSION
        if (spec.ptr_grad_idx_16) {
            delete [] spec.ptr_grad_idx_16;
        }
#endif

#if FP16_COMPRESSION
        if (spec.ptr_grad_val_16) {
            delete [] spec.ptr_grad_val_16;
        }
#endif

        {
            std::unique_lock<std::mutex> ul(delegate_delete_mutex_);

            bool is_last_sync = false;
            if (engine_->world_size() > 2) {
                if (delegate_sent_cnt_map_.find(spec.key) == delegate_sent_cnt_map_.end())
                    delegate_sent_cnt_map_[spec.key] = 0;

                if (delegate_sent_cnt_map_[spec.key] + 2 == engine_->world_size()) {
                    is_last_sync = true;
                    delegate_sent_cnt_map_.erase(spec.key);
                } else {
                    delegate_sent_cnt_map_[spec.key]++;
                }
            } else {
                is_last_sync = true;
            }

            if (is_last_sync) {
                const std::string rkey(spec.key + "!" + std::to_string(spec.iter));
                auto it = delegate_delete_map_.find(spec.key);
                if (it != delegate_delete_map_.end()) {
                    // meaning optimizer finished to use, so I delete
                    assert(it->second.first != nullptr);
                    assert(it->second.second != nullptr);
                    delete [] it->second.first;
                    delete [] it->second.second;
                    delegate_delete_map_.erase(it);
                } else {
                    // meaning optimizer didn't finished to use, so I need to wait
                    delegate_delete_map_[spec.key] = std::make_pair(nullptr, nullptr);
                }
            }
        }
    }
}

void CommManager::delegate_delete(const TrainTaskV2 *task) {
    std::unique_lock<std::mutex> ul(delegate_delete_mutex_);
    auto it = delegate_delete_map_.find(task->key());
    if (it != delegate_delete_map_.end()) {
        // meaning sender completed sending, so now I can delete
        assert(it->second.first == nullptr);
        assert(it->second.second == nullptr);
        delete [] task->compressed_grad_val_ptr_;
        delete [] task->compressed_grad_idx_ptr_;
        delegate_delete_map_.erase(it);
    } else {
        // meaning sender hasn't completed sending, so I need to wait
        delegate_delete_map_[task->key()] = std::make_pair(task->compressed_grad_val_ptr_, task->compressed_grad_idx_ptr_);
    }
}

void CommManager::pull_thread_main() {

    sigset_t mask;
    sigemptyset (&mask);
    sigaddset (&mask, SIGTERM);
    sigaddset (&mask, SIGHUP);
    pthread_sigmask(SIG_BLOCK, &mask, NULL);
    pthread_setname_np(pthread_self(), "CommPullThread");

    while (!finished_) {
        std::array<zmq::message_t, 4> recv_msgs; // possibility of double copy
        try {
            auto result = zmq::recv_multipart_n(sock_pull_, recv_msgs.data(), 4);
            assert(result.has_value() && result.value() == 4);

            std::shared_ptr<std::packaged_task<void ()>> ptr_callback_fn = nullptr;
            const std::string key(recv_msgs[0].to_string());
            RecvBufSpec recv_buf_spec;
            recv_buf_spec.receiver_ts = 0;
            bandwidth_monitor_.ingress(recv_msgs[0].size() + recv_msgs[1].size() + recv_msgs[2].size() + recv_msgs[3].size());
            uint8_t flag = recv_msgs[1].data<char>()[0];
            {
                std::unique_lock<std::mutex> ul(pull_callback_map_mutex_);
                if (pull_callback_map_.find(key) != pull_callback_map_.end()) {
                    auto &pair = pull_callback_map_[key];
                    ptr_callback_fn = pair.first;
                    recv_buf_spec = pair.second;
                    pull_callback_map_.erase(key);
                    // std::stringstream s;
                    // s << "[RECV] " << key << " : stored as pre-assigned, mapsz=" << pull_callback_map_.size();
                    // std::cerr << s.str() << std::endl;
                } else {
                    recv_buf_spec.is_malloc_segment = true;
                    recv_buf_spec.receiver_ts = get_timestamp();
                    recv_buf_spec.sender_ts = *reinterpret_cast<int64_t *>(recv_msgs[1].data());
#if FP16_COMPRESSION
                    if (flag & COMM_FLAG_FP16_VAL)
                        recv_buf_spec.len = recv_msgs[3].size() / sizeof(fp16_t);
                    else
                        recv_buf_spec.len = recv_msgs[3].size() / sizeof(float);
#else
                    recv_buf_spec.len = recv_msgs[3].size() / sizeof(float);
#endif


                    recv_buf_spec.idx = new uint32_t[recv_buf_spec.len * sizeof(float)];
#if IDX_COMPRESSION
                    if (flag & COMM_FLAG_UINT16_IDX) {
                        assert (recv_msgs[2].size() == recv_buf_spec.len * sizeof(uint16_t));
                        cast_uint16_to_uint32(recv_msgs[2].data<const uint16_t>(), recv_buf_spec.idx, recv_buf_spec.len);
                    } else {
                        assert (recv_msgs[2].size() == recv_buf_spec.len * sizeof(uint32_t));
                        memcpy(recv_buf_spec.idx, recv_msgs[2].data(), recv_msgs[2].size());
                    }
#else
                    assert (recv_msgs[2].size() == recv_buf_spec.len * sizeof(uint32_t));
                    memcpy(recv_buf_spec.idx, recv_msgs[2].data(), recv_msgs[2].size());
#endif

                    recv_buf_spec.val = new float[recv_buf_spec.len * sizeof(float)];
#if FP16_COMPRESSION
                    if (flag & COMM_FLAG_FP16_VAL) {
                        assert (recv_msgs[3].size() == recv_buf_spec.len * sizeof(fp16_t));
                        cast_fp16_to_fp32(recv_msgs[3].data<const fp16_t>(), recv_buf_spec.val, recv_buf_spec.len);
                    } else {
                        assert(recv_msgs[3].size() == recv_buf_spec.len * sizeof(float));
                        memcpy(recv_buf_spec.val, recv_msgs[3].data(), recv_msgs[3].size());
                    }
#else
                    assert(recv_msgs[3].size() == recv_buf_spec.len * sizeof(float));
                    memcpy(recv_buf_spec.val, recv_msgs[3].data(), recv_msgs[3].size());
#endif

                    pull_callback_map_[key] = std::make_pair(nullptr, recv_buf_spec);

                    // std::stringstream s;
                    // s << "[RECV] " << key << " : stored by new alloc, mapsz=" << pull_callback_map_.size();
                    // std::cerr << s.str() << std::endl;
                }
            }

            if (ptr_callback_fn) {
                std::string real_key = key.substr(0, key.find("!")); 
                // auto rcvd_iter = std::stoi(key.substr(key.find("!") + 1));
                // // here, iter may not be what it really wants
                // auto task = engine_->find_task_by_key(real_key);
                // if (task->grad_sync_iter() == rcvd_iter) {

                    assert(engine_);
                    assert(!recv_buf_spec.is_malloc_segment);
                    uint8_t flag = recv_msgs[1].data<char>()[0];

#if IDX_COMPRESSION
                    if (flag & COMM_FLAG_UINT16_IDX) {
                        assert (recv_msgs[2].size() == recv_buf_spec.len * sizeof(uint16_t));
                        cast_uint16_to_uint32(recv_msgs[2].data<const uint16_t>(), recv_buf_spec.idx, recv_buf_spec.len);
                    } else {
                        assert (recv_msgs[2].size() == recv_buf_spec.len * sizeof(uint32_t));
                        memcpy(recv_buf_spec.idx, recv_msgs[2].data(), recv_msgs[2].size());
                    }
#else
                    assert (recv_msgs[2].size() == recv_buf_spec.len * sizeof(uint32_t));
                    memcpy(recv_buf_spec.idx, recv_msgs[2].data(), recv_msgs[2].size());
#endif

#if FP16_COMPRESSION
                    if (flag & COMM_FLAG_FP16_VAL) {
                        assert (recv_msgs[3].size() == recv_buf_spec.len * sizeof(fp16_t));
                        cast_fp16_to_fp32(recv_msgs[3].data<const fp16_t>(), recv_buf_spec.val, recv_buf_spec.len);
                    } else {
                        assert(recv_msgs[3].size() == recv_buf_spec.len * sizeof(float));
                        memcpy(recv_buf_spec.val, recv_msgs[3].data(), recv_msgs[3].size());
                    }
#else
                    assert(recv_msgs[3].size() == recv_buf_spec.len * sizeof(float));
                    memcpy(recv_buf_spec.val, recv_msgs[3].data(), recv_msgs[3].size());
#endif

                    recv_buf_spec.sender_ts = *reinterpret_cast<int64_t *>(recv_msgs[1].data());
                    // std::cerr << "Receiving " << key << " ptr_grad_idx=" 
                    // << recv_buf_spec.idx  << " ptr_grad_val=" << recv_buf_spec.val << std::endl;

                    engine_->task_state_update_by_key(real_key, TASK_PENDING);
                    (*ptr_callback_fn)();


                // } else {
                //     std::cerr << "GOTCHA!!" << std::endl;
                // }
            }

            // if (recv_buf_spec.sender_ts != 0 && recv_buf_spec.receiver_ts != 0 && recv_buf_spec.len >= 1024 * 10) {
            //     sendStatReport(recv_buf_spec.sender_ts, recv_buf_spec.receiver_ts, recv_buf_spec.len * sizeof(float));
            // }


        } catch (zmq::error_t err) {
            std::cerr << err.what() << std::endl;
            std::cerr << err.num() << std::endl;
        }
    }
}

void CommManager::sendStatReport(const int64_t sender_timestamp, const int64_t receiver_timestamp, const uint32_t payload_size) {
    char buf[sizeof(MsgStatReport) + 1] = {0, };
    MsgStatReport *msg = reinterpret_cast<MsgStatReport *>(buf + 1);
    buf[0] = static_cast<char>(BrokerMsgType::STAT_REP);
    msg->sender_timestamp = sender_timestamp;
    msg->receiver_timestamp = receiver_timestamp;
    msg->payload_size = payload_size;
    // std::cerr << "Sending stat report, ts=" << msg->sender_timestamp << " -> " << msg->receiver_timestamp << ", bytes = " << msg->payload_size << std::endl;

    {
        std::unique_lock<std::mutex> ul(sock_mutex_);
        sock_ctrl_.send(zmq::buffer(buf, sizeof(MsgStatReport) + 1));
    }

    // recv dummy message here
    {
        std::unique_lock<std::mutex> ul(sock_mutex_);
        zmq::message_t recv_msg;
        auto result = sock_ctrl_.recv(recv_msg);
    }
}

void CommManager::initiateTimeSync() {

    char buf[sizeof(MsgStatReport) + 1] = {0, };
    MsgSyncTime *msg = reinterpret_cast<MsgSyncTime *>(buf + 1);
    buf[0] = static_cast<char>(BrokerMsgType::SYNC_TIME);
    
    for (int i = 0; i < 16; i++) {
        msg->timestamp = get_timestamp();
        {
            std::unique_lock<std::mutex> ul(sock_mutex_);
            sock_ctrl_.send(zmq::buffer(buf, sizeof(MsgStatReport) + 1));
        }

        // recv dummy message here
        {
            std::unique_lock<std::mutex> ul(sock_mutex_);
            zmq::message_t recv_msg;
            auto result = sock_ctrl_.recv(recv_msg);
        }   

        usleep(100*1000);
    }
}

void CommManager::sendInitmodel(const std::string skey, const float *ptr_model_val, const uint32_t len) {
    
    std::size_t buflen = 2 + skey.size() + len * sizeof(float);
    auto buf = new char[buflen];

    
    assert (skey.length() < 256);
    buf[0] = static_cast<char>(BrokerMsgType::MODEL_REP);
    buf[1] = skey.length();
    memcpy(buf + 2, skey.c_str(), skey.length());
    memcpy(buf + 2 + skey.length(), ptr_model_val, len * sizeof(float));
    
    std::unique_lock<std::mutex> ul(sock_mutex_);
    sock_ctrl_.send(zmq::buffer(buf, buflen));
    // std::cerr << "Sending Master Init Model for " << skey << std::endl;

    // recv dummy message here
    zmq::message_t recv_msg;
    auto result = sock_ctrl_.recv(recv_msg);
    delete [] buf;
}

RecvModelSpec CommManager::recvInitmodel(std::string skey) {
    // send request message here
    // std::cerr << "Requesting recv Init Model for " << skey << std::endl;
    std::unique_lock<std::mutex> ul(sock_mutex_);
    {
        char buf[2 + skey.length()] = {0,};
        buf[0] = static_cast<char>(BrokerMsgType::MODEL_REP);
        buf[1] = skey.length();
        strncpy(buf + 2, skey.c_str(), skey.length());
        sock_ctrl_.send(zmq::buffer(buf, 2 + skey.length()));
    }

    zmq::message_t recv_msg;    
    auto result = sock_ctrl_.recv(recv_msg);
    assert(result.has_value());

    // len is at offset 1
    assert(recv_msg.data<char>()[0] == static_cast<char>(BrokerMsgType::MODEL_REP));
    assert(recv_msg.size() >= 2);
    std::size_t key_len = static_cast<std::size_t>(recv_msg.data<char>()[1]);

    RecvModelSpec spec;
    spec.skey = std::string(recv_msg.data<char>() + 2, key_len);
    assert(recv_msg.size() > 2 + key_len);
    std::size_t payload_len = recv_msg.size() - 2 - key_len;
    assert(payload_len % sizeof(float) == 0);
    spec.ptr = std::make_unique<float []>(payload_len);
    spec.len = payload_len / sizeof(float);

    memcpy(spec.ptr.get(), recv_msg.data<char>() + 2 + key_len, payload_len);
    // std::cerr << "Received Init Model for " << skey << std::endl;

    return spec;
}
