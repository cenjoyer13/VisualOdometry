#include "FLANNMatcher.h"
#include <iostream>
#include <opencv2/core/ocl.hpp>

FLANNMatcher::FLANNMatcher(const OdometryConfig& cfg) : config(cfg) {
    // Extract parameters with safe fallbacks
    int trees = config.matcher_params.count("kdTrees") ? (int)config.matcher_params.at("kdTrees") : 5;
    int checks = config.matcher_params.count("searchChecks") ? (int)config.matcher_params.at("searchChecks") : 50;

    // Inject parameters into the FLANN Index and Search objects
    flann_matcher = cv::makePtr<cv::FlannBasedMatcher>(
        cv::makePtr<cv::flann::KDTreeIndexParams>(trees),
        cv::makePtr<cv::flann::SearchParams>(checks)
    );
}

std::vector<cv::DMatch> FLANNMatcher::match(DeviceBuffer& desc_old, 
                                            DeviceBuffer& desc_new,
                                            const std::vector<cv::KeyPoint>& kp_old,
                                            const std::vector<cv::KeyPoint>& kp_new) {
    if (kp_old.empty() || kp_new.empty()) return {};

    float ratio_thresh = config.matcher_params.count("ratio_thresh") ? config.matcher_params.at("ratio_thresh") : 0.75f;

    // ==========================================
    // 1. HARDWARE CAPABILITY RESOLVER
    // Strict Cascade: CUDA -> OPENCL -> CPU
    // ==========================================
    ComputeBackend target_backend = config.backend;

    if (target_backend == ComputeBackend::CUDA) {
        static bool warned_cuda = false;
        if (!warned_cuda) {
            std::cerr << "[FLANNMatcher] OpenCV lacks cv::cuda::FlannBasedMatcher. Cascading to OpenCL..." << std::endl;
            warned_cuda = true;
        }
        target_backend = ComputeBackend::OPENCL; // Fallback 1
    }

    if (target_backend == ComputeBackend::OPENCL) {
        if (!cv::ocl::haveOpenCL()) {
            static bool warned_ocl = false;
            if (!warned_ocl) {
                std::cerr << "[FLANNMatcher] OpenCL not available. Cascading to CPU..." << std::endl;
                warned_ocl = true;
            }
            target_backend = ComputeBackend::CPU; // Fallback 2
        } else {
            // Note: KD-Trees cause massive thread divergence on GPUs. 
            // OpenCV's T-API will safely process this on the CPU internally.
            static bool warned_flann_ocl = false;
            if (!warned_flann_ocl) {
                std::cerr << "[FLANNMatcher] Note: KD-Trees cannot be efficiently accelerated on GPUs. OpenCL T-API will process on CPU internally." << std::endl;
                warned_flann_ocl = true;
            }
        }
    }

    // ==========================================
    // 2. EXPLICIT EXECUTION ROUTING
    // ==========================================
    std::vector<std::vector<cv::DMatch>> knn_matches;

    if (target_backend == ComputeBackend::OPENCL) {
        
        // --- STRICT OPENCL EXECUTION ---
        cv::UMat u_old = desc_old.getAsOpenCL();
        cv::UMat u_new = desc_new.getAsOpenCL();

        // SAFETY CHECK: FLANN requires Float32 descriptors
        if (u_old.type() != CV_32F) u_old.convertTo(u_old, CV_32F);
        if (u_new.type() != CV_32F) u_new.convertTo(u_new, CV_32F);

        flann_matcher->knnMatch(u_old, u_new, knn_matches, 2);

    } else {
        
        // --- STRICT CPU EXECUTION ---
        cv::Mat cpu_old = desc_old.getAsCPU();
        cv::Mat cpu_new = desc_new.getAsCPU();

        if (cpu_old.type() != CV_32F) cpu_old.convertTo(cpu_old, CV_32F);
        if (cpu_new.type() != CV_32F) cpu_new.convertTo(cpu_new, CV_32F);

        // Maximize OpenCV internal TBB/OpenMP threading for CPU execution
        cv::setNumThreads(config.num_threads); 
        flann_matcher->knnMatch(cpu_old, cpu_new, knn_matches, 2);
    }

    // ==========================================
    // 3. LOWE'S RATIO TEST
    // ==========================================
    std::vector<cv::DMatch> good_matches;
    good_matches.reserve(knn_matches.size());

    for (const auto& match_pair : knn_matches) {
        if (match_pair.size() < 2) continue;
        
        const cv::DMatch& best = match_pair[0];
        const cv::DMatch& second_best = match_pair[1];

        if (best.distance < ratio_thresh * second_best.distance) {
            good_matches.push_back(best);
        }
    }
    
    return good_matches;
}
