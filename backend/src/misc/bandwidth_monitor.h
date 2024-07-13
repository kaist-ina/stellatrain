#ifndef BANDWIDTH_MONITOR_H
#define BANDWIDTH_MONITOR_H

#include <iostream>
#include <deque>
#include <chrono>

class BandwidthMonitor {
public:
    BandwidthMonitor(
        std::chrono::milliseconds window_size = std::chrono::milliseconds(1000), 
        std::chrono::milliseconds sliding_window = std::chrono::milliseconds(100)) : window_size_(window_size), 
                         sliding_window_(sliding_window) {}

    void ingress(int packet_size) {
        auto now = std::chrono::steady_clock::now();
        packet_data.push_back({now, packet_size});
        
        // Remove packets that are outside of the window size
        while (!packet_data.empty() && now - packet_data.front().first >= window_size_) {
            packet_data.pop_front();
        }
    }

    double query_bandwidth() {
        auto now = std::chrono::steady_clock::now();

        // Removing packets that are older than the window size
        while (!packet_data.empty() && now - packet_data.front().first >= window_size_) {
            packet_data.pop_front();
        }

        const bool mode_max = true;

        if (mode_max) {
            
            std::chrono::steady_clock::time_point time;
            
            int windows[window_size_ / sliding_window_ + 1] = {0, };

            for (const auto& packet : packet_data) {
                auto time = packet.first;
                auto packet_size = packet.second;
                int window_number = (now - time) / sliding_window_;
                windows[window_number] += packet_size;
            }

            // find max
            int max_bytes = 0;
            for (int i = 0; i < window_size_ / sliding_window_ + 1; i++) {
                max_bytes = std::max(max_bytes, windows[i]);
            }
            
            // Getting the duration in seconds to calculate throughput in bytes per second
            double duration_in_seconds = std::chrono::duration<double>(sliding_window_).count();

            return max_bytes / duration_in_seconds;


        } else {
            int total_bytes = 0;
            for (const auto& packet : packet_data) {
                total_bytes += packet.second;
            }

            // Getting the duration in seconds to calculate throughput in bytes per second
            double duration_in_seconds = std::chrono::duration<double>(window_size_).count();

            return total_bytes / duration_in_seconds;
        }
    }

    double query_bandwidth_mbps() {
        return query_bandwidth() * 8 / 1000000;
    }

private:
    using TimePoint = std::chrono::steady_clock::time_point;

    std::chrono::milliseconds window_size_;
    std::chrono::milliseconds sliding_window_;

    std::deque<std::pair<TimePoint, int>> packet_data;
};

#endif