#include "ORBDetector.h"
#include <iostream>

// Needed for OpenCL T-API status checks
#include <opencv2/core/ocl.hpp> 

#ifdef HAS_CUDA
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudafeatures2d.hpp>
#endif

ORBDetector::ORBDetector(const OdometryConfig& cfg) : config(cfg) {
    // Safely extract parameters from the config dictionary
    int num_features = config.detector_params.count("num_features") ? (int)config.detector_params.at("num_features") : 500;
    float scale_factor = config.detector_params.count("scale_factor") ? config.detector_params.at("scale_factor") : 1.2f;
    int nlevels = config.detector_params.count("nlevels") ? (int)config.detector_params.at("nlevels") : 8;

    // 1. Always initialize the CPU/OpenCL fallback object
    orb_cpu = cv::ORB::create(num_features, scale_factor, nlevels);

    // 2. Initialize CUDA object if compiled with support AND requested
#ifdef HAS_CUDA
    if (config.backend == ComputeBackend::CUDA) {
        orb_cuda = cv::cuda::ORB::create(num_features, scale_factor, nlevels);
    }
#endif
}

void ORBDetector::detect(DeviceBuffer& image, 
                         std::vector<cv::KeyPoint>& out_keypoints, 
                         DeviceBuffer& out_descriptors) {
    
    switch (config.backend) {

        case ComputeBackend::CUDA: {
#ifdef HAS_CUDA
            // Retrieve or upload the image to VRAM
            cv::cuda::GpuMat* d_img = static_cast<cv::cuda::GpuMat*>(image.getCUDAPointer());
            
            // We need a grayscale image for ORB
            cv::cuda::GpuMat d_gray;
            if (d_img->channels() == 3) {
                cv::cuda::cvtColor(*d_img, d_gray, cv::COLOR_BGR2GRAY);
            } else {
                d_gray = *d_img;
            }

            cv::cuda::GpuMat d_descriptors;
            orb_cuda->detectAndComputeAsync(d_gray, cv::cuda::GpuMat(), out_keypoints, d_descriptors);

            // Wrap the resulting GPU memory into the output buffer
            out_descriptors = DeviceBuffer(d_descriptors); 
            break;
#else
            std::cerr << "[ORBDetector] WARNING: CUDA requested but not compiled. Falling back to CPU." << std::endl;
            // Fallthrough to CPU intentionally
#endif
        }

        case ComputeBackend::OPENCL: {
            // Check if OpenCL is actually available on this machine
            if (!cv::ocl::haveOpenCL()) {
                std::cerr << "[ORBDetector] WARNING: OpenCL not available. Falling back to CPU." << std::endl;
                // Will gracefully run as standard CPU Mat inside the T-API
            }

            // getAsOpenCL() triggers a copy to a cv::UMat. 
            // Passing a UMat to OpenCV functions automatically utilizes OpenCL.
            cv::UMat u_img = image.getAsOpenCL();
            cv::UMat u_gray;
            
            if (u_img.channels() == 3) {
                cv::cvtColor(u_img, u_gray, cv::COLOR_BGR2GRAY);
            } else {
                u_gray = u_img;
            }

            cv::UMat u_descriptors;
            orb_cpu->detectAndCompute(u_gray, cv::noArray(), out_keypoints, u_descriptors);

            // Wrap the resulting OpenCL memory into the output buffer
            out_descriptors = DeviceBuffer(u_descriptors);
            break;
        }

        case ComputeBackend::CPU:
        default: {
            // Strictly CPU-bound execution
            cv::Mat cpu_img = image.getAsCPU();
            cv::Mat gray_img;
            
            if (cpu_img.channels() == 3) {
                cv::cvtColor(cpu_img, gray_img, cv::COLOR_BGR2GRAY);
            } else {
                gray_img = cpu_img;
            }

            cv::Mat cpu_descriptors;
            orb_cpu->detectAndCompute(gray_img, cv::noArray(), out_keypoints, cpu_descriptors);

            // Wrap the resulting RAM memory into the output buffer
            out_descriptors = DeviceBuffer(cpu_descriptors);
            break;
        }
    }
}
