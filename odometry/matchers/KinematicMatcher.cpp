#include "KinematicMatcher.h"
#include <iostream>

#ifdef HAS_CUDA
#include <opencv2/cudafeatures2d.hpp>
#endif

KinematicMatcher::KinematicMatcher(const OdometryConfig& cfg) : config(cfg) {
    // Create the CPU matcher (NORM_HAMMING is required for ORB's binary descriptors)
    matcher_cpu = cv::BFMatcher::create(cv::NORM_HAMMING);

#ifdef HAS_CUDA
    if (config.backend == ComputeBackend::CUDA) {
        matcher_cuda = cv::cuda::DescriptorMatcher::createBFMatcher(cv::NORM_HAMMING);
    }
#endif
}

std::vector<cv::DMatch> KinematicMatcher::match(DeviceBuffer& desc_old, 
                                                DeviceBuffer& desc_new,
                                                const std::vector<cv::KeyPoint>& kp_old,
                                                const std::vector<cv::KeyPoint>& kp_new) {
    
    // Safety check: Cannot match if one of the descriptor sets is empty
    if (kp_old.empty() || kp_new.empty()) {
        return {};
    }

    // Extract parameters from config (or use your original defaults)
    float ratio_thresh = config.matcher_params.count("ratio_thresh") ? config.matcher_params.at("ratio_thresh") : 0.75f;
    float max_hamming = config.matcher_params.count("max_hamming") ? config.matcher_params.at("max_hamming") : 50.0f;
    float max_pixel_jump = config.matcher_params.count("max_pixel_jump") ? config.matcher_params.at("max_pixel_jump") : 80.0f;

    // k=2 finds the best and second-best matches (required for the ratio test)
    std::vector<std::vector<cv::DMatch>> knn_matches;

    // 1. HARDWARE-ACCELERATED MATCHING
    switch (config.backend) {
        case ComputeBackend::CUDA: {
#ifdef HAS_CUDA
            cv::cuda::GpuMat* d_old = static_cast<cv::cuda::GpuMat*>(desc_old.getCUDAPointer());
            cv::cuda::GpuMat* d_new = static_cast<cv::cuda::GpuMat*>(desc_new.getCUDAPointer());
            matcher_cuda->knnMatch(*d_old, *d_new, knn_matches, 2);
            break;
#else
            std::cerr << "[KinematicMatcher] WARNING: CUDA requested but not compiled. Falling back to CPU." << std::endl;
            // Fallthrough
#endif
        }
        case ComputeBackend::OPENCL: {
            // Passing OpenCL UMat memory triggers the Transparent API
            cv::UMat u_old = desc_old.getAsOpenCL();
            cv::UMat u_new = desc_new.getAsOpenCL();
            matcher_cpu->knnMatch(u_old, u_new, knn_matches, 2);
            break;
        }
        case ComputeBackend::CPU:
        default: {
            cv::Mat cpu_old = desc_old.getAsCPU();
            cv::Mat cpu_new = desc_new.getAsCPU();
            matcher_cpu->knnMatch(cpu_old, cpu_new, knn_matches, 2);
            break;
        }
    }

    // 2. KINEMATIC & RATIO FILTERING (Always on CPU, as keypoints are lightweight)
    std::vector<cv::DMatch> good_matches;
    good_matches.reserve(knn_matches.size()); // Reserve memory to prevent reallocations

    for (const auto& match_pair : knn_matches) {
        // Ensure we actually found 2 matches to compare
        if (match_pair.size() < 2) continue;

        const cv::DMatch& best = match_pair[0];
        const cv::DMatch& second_best = match_pair[1];

        // 1st Filter: Lowe's Ratio Test + Absolute Hamming Threshold
        if (best.distance < ratio_thresh * second_best.distance && best.distance < max_hamming) {
            
            // 2nd Filter: Kinematic Bounds (Your 80.0 pixel limit)
            const cv::Point2f& pt_old = kp_old[best.queryIdx].pt;
            const cv::Point2f& pt_new = kp_new[best.trainIdx].pt;
            
            double pixel_movement = cv::norm(pt_new - pt_old);

            if (pixel_movement < max_pixel_jump) {
                good_matches.push_back(best);
            }
        }
    }

    return good_matches;
}
