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
    SYNCED      // Data is identical on both devices
};

class DeviceBuffer {
private:
    BufferLocation location;
    cv::Mat cpu_mat;
    
    // We use a void pointer with a custom deleter to hold the cv::cuda::GpuMat 
    // without requiring <opencv2/core/cuda.hpp> in this header file.
    // This keeps the rest of the project compilation clean.
    std::shared_ptr<void> gpu_mat_ptr; 
    
    cv::UMat opencl_mat; 

public:
    DeviceBuffer() : location(BufferLocation::CPU_ONLY) {}

    // Constructor for initial CPU data (e.g., from AirSim)
    explicit DeviceBuffer(const cv::Mat& cpu_data) 
        : cpu_mat(cpu_data.clone()), location(BufferLocation::CPU_ONLY) {}

    BufferLocation getLocation() const { return location; }

    // =========================================================
    // LAZY SYNCHRONIZATION METHODS
    // =========================================================

    // Returns CPU memory. Downloads from GPU ONLY if necessary.
    cv::Mat& getAsCPU() {
        if (location == BufferLocation::GPU_ONLY) {
            // Logic to download from GPU to cpu_mat happens here
            // (Implemented in the .cpp file where CUDA headers are included)
            location = BufferLocation::SYNCED;
        }
        return cpu_mat;
    }

    // Returns OpenCV Transparent API memory (AMD/Intel fallback)
    cv::UMat& getAsOpenCL() {
        if (location == BufferLocation::CPU_ONLY) {
            cpu_mat.copyTo(opencl_mat);
            location = BufferLocation::SYNCED;
        }
        return opencl_mat;
    }

    // A getter for the CUDA pointer, cast locally by the GPU modules
    void* getCUDAPointer() {
        return gpu_mat_ptr.get();
    }
    
    // Methods to allocate and upload to CUDA (implemented in .cpp)
    void uploadToCUDA(); 
};


struct CameraIntrinsics {
    float fx, fy, cx, cy;
};

// The Master Configuration Object
struct OdometryConfig {
    ComputeBackend backend = ComputeBackend::CPU;
    CameraIntrinsics intrinsics;
    
    // Generic dictionaries for module-specific tuning
    std::map<std::string, float> detector_params;
    std::map<std::string, float> matcher_params;
    std::map<std::string, float> pose_params;
};

// Ground Truth State from AirSim
struct GroundTruthData {
    cv::Vec3f position;    // True X, Y, Z
    cv::Vec3f orientation; // True Pitch, Roll, Yaw
};
