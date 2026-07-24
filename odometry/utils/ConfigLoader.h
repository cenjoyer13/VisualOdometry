#pragma once
#include <opencv2/core.hpp>
#include <string>
#include "../OdometryTypes.h"
#include "../camera/CameraModelConfig.h"

// Per-folder inputs for OdomLogEvaluator: a bare folder of frame_NNNNNN.jpg
// images with no ground truth. times_file is optional (a "filename,timestamp"
// CSV, as logged alongside the images); when present it enables start_time/
// end_time filtering (seconds elapsed since the first logged timestamp,
// matching RosbagConfig's convention). Without times_file, start_time/
// end_time are ignored and every frame in the folder is processed.
struct ImageSequenceConfig {
    std::string path;
    std::string times_file;
    // Optional MAVLink GLOBAL_POSITION_INT_log.csv logged alongside the images.
    // When present the evaluator converts its GPS fixes to a local-tangent-plane
    // ENU position per frame, enabling a metric AirSim scale estimator instead
    // of the unitless fallback. Empty = no ground truth.
    std::string ground_truth_file;
    double start_time = 0.0;
    double end_time   = -1.0;   // < 0 means unbounded
    double viz_scale  = 0.5;                   // 2D trajectory visualizer px-per-metre scale
    double aligner_init_distance = 10.0;       // GT travel (m) before solving the VO->GT yaw
    cv::Vec3d traj_sign{1.0, 1.0, 1.0};        // per-axis sign flip for the VO trajectory (handedness)
    cv::Mat body_T_cam0;                        // 4x4 cam0->body extrinsic; empty = identity
    // Feed GLOBAL_POSITION_INT.relative_alt as AGL so a Homography pose
    // estimator scales its raw t/d by the plane distance instead of consuming a
    // GT position delta. Same key/meaning as RosbagConfig::altimeter_scale.
    bool altimeter_scale = false;
};

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
    cv::Mat body_T_cam0;                     // 4x4 cam0->body extrinsic; empty = identity
    cv::Vec3d traj_sign{1.0, 1.0, 1.0};      // per-axis sign flip for the logged trajectory
    double viz_scale = 0.5;                   // 2D trajectory visualizer px-per-metre scale
    double aligner_init_distance = 10.0;      // GT travel (m) before solving the VO->GT yaw
    bool altimeter_scale = false;             // feed AGL so homography scale comes from altitude
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

    // OdomLogEvaluator helper: image_sequence.* (folder of frame_NNNNNN.jpg,
    // no ground truth). Returns false if the block or path key is absent.
    bool loadImageSequenceConfig(ImageSequenceConfig& out_config);

    // PathPlayer helper: airsim.playback_velocity. Returns false if absent.
    bool loadPlaybackVelocity(float& out_velocity);

    // RosbagEvaluator helper: rosbag.* block plus the top-level body_T_cam0
    // extrinsic. Returns false if the rosbag block is absent or has no bag_path.
    bool loadRosbagConfig(RosbagConfig& out_config);

    // Reads the camera.model_type block into a CameraModelConfig. Returns false
    // when no model_type is present, signalling the caller to keep the plain
    // fx/fy/cx/cy intrinsics from loadOdometryConfig and skip undistortion.
    bool loadCameraModel(CameraModelConfig& out_config);

private:
    cv::FileStorage fs;
};
