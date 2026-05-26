#pragma once
#include <opencv2/core.hpp>
#include <memory>
#include <map>
#include <string>

// Enum for hardware acceleration routing
enum class ComputeBackend {
    CPU,
    CUDA,    
    OPENCL   
};

// State tracker for our memory abstraction
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
        // Automatically sync to CPU memory as well
        cpu_mat = opencl_mat.getMat(cv::ACCESS_READ).clone(); 
    }

    BufferLocation getLocation() const { return location; }

    // =========================================================
    // LAZY SYNCHRONIZATION METHODS
    // =========================================================

    // Declared here, implemented in OdometryTypes.cpp to hide CUDA headers
    cv::Mat& getAsCPU();
    void uploadToCUDA(); 

    // OpenCL fallback uses Transparent API, safe to inline
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

// All knobs exposed by LocalBundleAdjustment. Defaults match the values that
// were previously hardcoded in LocalBundleAdjustment.cpp.
struct LBAParams {
    // Sliding-window mechanics
    int window_size = 10;
    int opt_stride = 2;

    // Tight prior on the oldest pose in the window — pins the gauge.
    double anchor_prior_sigma = 1e-4;

    // Loose prior on the newest pose — caps the cumulative drift the
    // SmartFactor chain is allowed to pull within one optimization.
    double end_prior_rot_sigma = 0.02;     // rad, per axis
    double end_prior_trans_sigma = 0.20;   // m,  per axis

    // BetweenFactor noise for moving frames.
    double between_rot_sigma = 0.02;
    double between_trans_sigma = 0.05;

    // BetweenFactor noise for stationary frames (identity measurement).
    double stationary_rot_sigma = 0.005;
    double stationary_trans_sigma = 0.01;

    // SmartProjectionPoseFactor settings.
    double pixel_sigma = 1.0;              // pixel noise sigma
    double rank_tolerance = 1e-5;
    double outlier_threshold = 3.0;        // post-triangulation reprojection cutoff (px)

    // Per-track admission filter.
    int min_observations = 4;
    double min_bbox_diagonal = 25.0;       // pixel parallax floor

    // Correction publishing acceptance.
    int min_smart_factors = 50;
    double max_correction_translation = 0.5;  // m
    double max_correction_rotation_deg = 5.0;
};

struct OdometryConfig {
    ComputeBackend backend = ComputeBackend::CPU;
    CameraIntrinsics intrinsics;

    std::string detector_type = "SIFT";
    std::string matcher_type = "FLANN";

    int num_threads = 1;
    bool verbose = false;          // Global debug toggle (YAML system.verbose or CLI --debug).

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
};

struct PipelineMetrics {
    double time_detect_ms = 0.0;
    double time_match_ms = 0.0;
    double time_pose_ms = 0.0;
    double time_total_ms = 0.0;
    double fps = 0.0;
};
