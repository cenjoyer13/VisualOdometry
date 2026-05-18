#pragma once
#include <mutex>
#include <atomic>
#include <opencv2/core.hpp>

// Cleaned up input struct (No YOLO/Tracking flags)
struct DroneInput {
    float vx = 0.0f;
    float vy = 0.0f;
    float vz = 0.0f;
    float yaw = 0.0f;
    bool enable_odometry = false;
};

// Thread-safe context for UI <-> Worker communication
struct SharedContext {
    DroneInput input;
    cv::Mat last_frame;
    std::mutex data_mutex;
    std::atomic<bool> has_new_frame{false};
    std::atomic<bool> is_running{true};
};

void runDroneLogic(SharedContext* ctx);
