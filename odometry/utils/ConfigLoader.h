#pragma once
#include <opencv2/core.hpp>
#include <string>
#include "../OdometryTypes.h"

// Per-bag inputs for RosbagEvaluator. All fields ship with sane defaults that
// match the Python RosbagLoader; bag_path is the only mandatory key.
struct RosbagConfig {
    std::string bag_path;
    std::string img_topic = "/camera/image_mono";
    std::string gps_topic = "/fix";
    std::string imu_topic = "/imu/data";
    std::string ppk_path;                   // empty disables PPK overlay
    double start_time = 0.0;                // seconds from bag start
    double end_time   = -1.0;               // < 0 means unbounded
};

// Wraps an OpenCV FileStorage handle and converts a YAML config file into an
// OdometryConfig. Fields absent from the YAML are left untouched on the
// out-parameter, so callers pre-seed defaults (per-dataset intrinsics,
// fallback backend, etc.) before invoking loadOdometryConfig().
class ConfigLoader {
public:
    explicit ConfigLoader(const std::string& yaml_path);

    bool isOpen() const { return fs.isOpened(); }

    // Populates detector/matcher/bucketing/LBA/system blocks and camera
    // intrinsics. Returns false only if the file failed to open.
    bool loadOdometryConfig(OdometryConfig& out_config);

    // KittiEvaluator helper: dataset.root_path / dataset.sequence.
    bool loadKittiDataset(std::string& out_root_path, std::string& out_sequence);

    // PathPlayer helper: airsim.playback_velocity. Returns false if absent.
    bool loadPlaybackVelocity(float& out_velocity);

    // RosbagEvaluator helper: rosbag.* block. Returns false if the block is
    // absent or has no bag_path.
    bool loadRosbagConfig(RosbagConfig& out_config);

private:
    cv::FileStorage fs;
};
