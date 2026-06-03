#pragma once
#include <opencv2/core.hpp>
#include <memory>
#include <map>
#include <string>

#include "inertial/ImuTypes.h"

// Backend routing tag carried on every detector/matcher.
enum class ComputeBackend {
    CPU,
    CUDA,
    OPENCL
};

// DeviceBuffer residency state.
enum class BufferLocation {
    CPU_ONLY,
    GPU_ONLY,
    SYNCED
};

class DeviceBuffer {
private:
    BufferLocation location;
    cv::Mat cpu_mat;
    std::shared_ptr<void> gpu_mat_ptr; 
    cv::UMat opencl_mat; 

public:
    DeviceBuffer() : location(BufferLocation::CPU_ONLY) {}

    explicit DeviceBuffer(const cv::Mat& cpu_data) 
        : cpu_mat(cpu_data.clone()), location(BufferLocation::CPU_ONLY) {}
        
    explicit DeviceBuffer(const cv::UMat& ocl_data)
        : opencl_mat(ocl_data.clone()), location(BufferLocation::SYNCED) {
        // Eager mirror to host memory; OpenCL UMat construction is rare enough
        // that the upfront copy keeps later getAsCPU calls O(1).
        cpu_mat = opencl_mat.getMat(cv::ACCESS_READ).clone();
    }

    BufferLocation getLocation() const { return location; }

    // Lazy synchronisation accessors.

    // Out-of-line so CUDA headers stay out of OdometryTypes.h.
    cv::Mat& getAsCPU();
    void uploadToCUDA();

    // OpenCL accessor is inlined: the T-API path has no CUDA dependency.
    cv::UMat& getAsOpenCL() {
        if (location == BufferLocation::CPU_ONLY) {
            cpu_mat.copyTo(opencl_mat);
            location = BufferLocation::SYNCED;
        }
        return opencl_mat;
    }

    void* getCUDAPointer() {
        return gpu_mat_ptr.get();
    }
};

struct CameraIntrinsics {
    float fx, fy, cx, cy;
};

struct BucketingConfig {
    bool enabled = false;
    int grid_cols = 10;
    int grid_rows = 10;
    int max_features_per_bucket = 20;
};

// LocalBundleAdjustment tunables. Defaults are the values previously
// hardcoded inside runOptimization; YAML overrides each field individually.
struct LBAParams {
    // Sliding-window mechanics.
    int window_size = 10;
    int opt_stride = 2;

    // Tight prior on the oldest pose in the window; pins the gauge so the
    // optimizer cannot drift the window globally.
    double anchor_prior_sigma = 1e-4;

    // Loose prior on the newest pose; caps how far one pass is allowed to
    // pull the trajectory before publishing.
    double end_prior_rot_sigma = 0.02;     // rad, per axis
    double end_prior_trans_sigma = 0.20;   // m, per axis

    // BetweenFactor noise on consecutive moving frames.
    double between_rot_sigma = 0.02;
    double between_trans_sigma = 0.05;

    // Tighter BetweenFactor noise when the relative measurement is identity
    // (stationary frame): trusts the no-motion observation more.
    double stationary_rot_sigma = 0.005;
    double stationary_trans_sigma = 0.01;

    // SmartProjectionPoseFactor settings.
    double pixel_sigma = 1.0;              // pixel noise sigma
    double rank_tolerance = 1e-5;
    double outlier_threshold = 3.0;        // post-triangulation reprojection cutoff (px)

    // Per-track admission filter: rejects tracks with too few observations
    // or too little parallax to triangulate stably.
    int min_observations = 4;
    double min_bbox_diagonal = 25.0;       // pixel parallax floor

    // Correction-publishing gates. Optimizations that fall below
    // min_smart_factors or produce a jump larger than the caps are dropped.
    int min_smart_factors = 50;
    double max_correction_translation = 0.5;  // m
    double max_correction_rotation_deg = 5.0;
};

struct OdometryConfig {
    ComputeBackend backend = ComputeBackend::CPU;
    CameraIntrinsics intrinsics;

    std::string detector_type = "SIFT";
    std::string matcher_type = "FLANN";
    std::string pose_estimator_type = "Essential";   // "Essential" | "Homography"

    // Keyframing caps. The frontend anchor is held while parallax accumulates;
    // a keyframe is forced (to keep matching alive) once matches drop below
    // keyframe_min_matches or the anchor has been held keyframe_max_skip frames.
    int keyframe_max_skip = 20;
    int keyframe_min_matches = 30;

    // Optional IMU layer (default ImuMode::Off = vision-only, unchanged).
    ImuParams imu_params;

    int num_threads = 1;
    // Global debug toggle, set by YAML system.verbose or CLI --debug.
    bool verbose = false;

    bool use_local_ba = false;
    LBAParams lba_params;

    BucketingConfig bucketing_params;

    std::map<std::string, float> detector_params;
    std::map<std::string, float> matcher_params;
    std::map<std::string, float> pose_params;
};

struct GroundTruthData {
    cv::Vec3f position;
    cv::Vec3f orientation;
    float altitude = -1.0f;   // AGL (m); <= 0 means not provided (NaN unusable
                              // here: the -ffast-math build folds isnan to false)
};

struct PipelineMetrics {
    double time_detect_ms = 0.0;
    double time_match_ms = 0.0;
    double time_pose_ms = 0.0;
    double time_total_ms = 0.0;
    double fps = 0.0;
};
