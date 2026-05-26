#pragma once
#include <opencv2/core.hpp>
#include <string>
#include "../OdometryTypes.h"

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

private:
    cv::FileStorage fs;
};
