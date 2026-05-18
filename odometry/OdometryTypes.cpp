#include "OdometryTypes.h"
#include <iostream>

#ifdef HAS_CUDA
#include <opencv2/core/cuda.hpp>
#endif

// Custom deleter for our void pointer to safely destroy cv::cuda::GpuMat
// without exposing the CUDA type to the rest of the project.
void cudaMatDeleter(void* ptr) {
#ifdef HAS_CUDA
    delete static_cast<cv::cuda::GpuMat*>(ptr);
#endif
}

void DeviceBuffer::uploadToCUDA() {
#ifdef HAS_CUDA
    if (location == BufferLocation::GPU_ONLY || location == BufferLocation::SYNCED) {
        return; // Already in VRAM
    }
    
    cv::cuda::GpuMat* d_mat = new cv::cuda::GpuMat();
    d_mat->upload(cpu_mat);
    
    // Assign to shared_ptr with our custom deleter
    gpu_mat_ptr = std::shared_ptr<void>(d_mat, cudaMatDeleter);
    location = BufferLocation::SYNCED;
#else
    std::cerr << "[DeviceBuffer] ERROR: uploadToCUDA called but HAS_CUDA is not defined." << std::endl;
#endif
}

cv::Mat& DeviceBuffer::getAsCPU() {
    if (location == BufferLocation::GPU_ONLY) {
#ifdef HAS_CUDA
        if (gpu_mat_ptr) {
            cv::cuda::GpuMat* d_mat = static_cast<cv::cuda::GpuMat*>(gpu_mat_ptr.get());
            d_mat->download(cpu_mat);
        }
#endif
        location = BufferLocation::SYNCED;
    }
    // Note: If data is in OpenCL (cv::UMat), OpenCV's Transparent API handles 
    // the synchronization to cv::Mat automatically under the hood.
    return cpu_mat;
}
