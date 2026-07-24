#pragma once
#include <mutex>
#include <atomic>
#include <opencv2/core.hpp>

#include "../odometry/OdometryTypes.h"

// Keyboard-derived velocity command from the UI thread.
struct DroneInput {
    float vx = 0.0f;
    float vy = 0.0f;
    float vz = 0.0f;
    float yaw = 0.0f;
    bool enable_odometry = false;
};

// Thread-safe channel between the UI thread (main.cpp) and the worker
// thread (runDroneLogic). The UI writes `input`, the worker writes
// `last_frame` + `has_new_frame`. `is_running` is the kill switch.
struct SharedContext {
    DroneInput input;
    cv::Mat last_frame;
    std::mutex data_mutex;
    std::atomic<bool> has_new_frame{false};
    std::atomic<bool> is_running{true};

    // Populated by main() before the worker thread is spawned.
    OdometryConfig config;
    std::string log_path = "freeplay_log.csv";
};

void runDroneLogic(SharedContext* ctx);
