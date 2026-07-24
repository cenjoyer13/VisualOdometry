#include "OdometryTypes.h"
#include <iostream>

#ifdef HAS_CUDA
#include <opencv2/core/cuda.hpp>
#endif

// Deleter for the type-erased GpuMat pointer; lets DeviceBuffer hold a
// shared_ptr<void> without leaking the CUDA type into the header.
void cudaMatDeleter(void* ptr) {
#ifdef HAS_CUDA
    delete static_cast<cv::cuda::GpuMat*>(ptr);
#endif
}

void DeviceBuffer::uploadToCUDA() {
#ifdef HAS_CUDA
    if (location == BufferLocation::GPU_ONLY || location == BufferLocation::SYNCED) {
        return;
    }

    cv::cuda::GpuMat* d_mat = new cv::cuda::GpuMat();
    d_mat->upload(cpu_mat);

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
    // OpenCL UMat path: the T-API materialises cv::Mat from the UMat on
    // demand, so no explicit synchronisation is needed here.
    return cpu_mat;
}
