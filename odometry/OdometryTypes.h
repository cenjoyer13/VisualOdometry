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


struct OdometryConfig {
    ComputeBackend backend = ComputeBackend::CPU;
    CameraIntrinsics intrinsics;
    
    std::string detector_type = "SIFT"; 
    std::string matcher_type = "FLANN"; 

    int num_threads = 1; // <-- NEW: Global thread limit
    
    bool use_local_ba = false;
    int lba_window_size = 10;
    
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
