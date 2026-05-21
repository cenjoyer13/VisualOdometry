#include "SIFTDetector.h"
#include "../utils/FeatureUtils.h"
#include <iostream>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/ocl.hpp>

SIFTDetector::SIFTDetector(const OdometryConfig& cfg) : config(cfg) {
    nfeatures = config.detector_params.count("nfeatures") ? (int)config.detector_params.at("nfeatures") : 3000;
    nOctaveLayers = config.detector_params.count("nOctaveLayers") ? (int)config.detector_params.at("nOctaveLayers") : 3;
    contrastThreshold = config.detector_params.count("contrastThreshold") ? config.detector_params.at("contrastThreshold") : 0.04;
    edgeThreshold = config.detector_params.count("edgeThreshold") ? config.detector_params.at("edgeThreshold") : 10.0;
    sigma = config.detector_params.count("sigma") ? config.detector_params.at("sigma") : 1.6;

    sift = cv::SIFT::create(nfeatures, nOctaveLayers, contrastThreshold, edgeThreshold, sigma);
}

void SIFTDetector::detect(DeviceBuffer& image, std::vector<cv::KeyPoint>& out_keypoints, DeviceBuffer& out_descriptors) {
    
    // ==========================================
    // 1. HARDWARE CAPABILITY RESOLVER
    // Strict Cascade: CUDA -> OPENCL -> CPU
    // ==========================================
    ComputeBackend target_backend = config.backend;

    if (target_backend == ComputeBackend::CUDA) {
        static bool warned_cuda = false;
        if (!warned_cuda) {
            std::cerr << "[SIFTDetector] OpenCV lacks cv::cuda::SIFT. Cascading to OpenCL..." << std::endl;
            warned_cuda = true;
        }
        target_backend = ComputeBackend::OPENCL;
    }

    if (target_backend == ComputeBackend::OPENCL) {
        if (!cv::ocl::haveOpenCL()) {
            static bool warned_ocl = false;
            if (!warned_ocl) {
                std::cerr << "[SIFTDetector] OpenCL not available. Cascading to CPU..." << std::endl;
                warned_ocl = true;
            }
            target_backend = ComputeBackend::CPU;
        }
    }

    // ==========================================
    // 2. LOGIC-FEATURE OVERRIDE
    // Bucketing strictly requires CPU multithreading
    // ==========================================
    if (config.bucketing_params.enabled && target_backend != ComputeBackend::CPU) {
        static bool warned_bucketing = false;
        if (!warned_bucketing) {
            std::cerr << "[SIFTDetector] True spatial bucketing requested. Bypassing GPU to prevent PCIe bottlenecks. Forcing CPU Multithreading..." << std::endl;
            warned_bucketing = true;
        }
        target_backend = ComputeBackend::CPU; // Hard override
    }


    // ==========================================
    // 3. EXPLICIT EXECUTION ROUTING
    // ==========================================
    if (target_backend == ComputeBackend::OPENCL) {
        
        // --- STRICT OPENCL EXECUTION (Whole Image Only) ---
        cv::UMat u_img = image.getAsOpenCL();
        cv::UMat u_gray;
        
        if (u_img.channels() == 3) {
            cv::cvtColor(u_img, u_gray, cv::COLOR_BGR2GRAY);
        } else {
            u_gray = u_img;
        }

        cv::UMat u_descriptors;
        sift->detectAndCompute(u_gray, cv::noArray(), out_keypoints, u_descriptors);
        out_descriptors = DeviceBuffer(u_descriptors);
        
    } else {
        
        // --- STRICT CPU EXECUTION (Bucketed or Standard) ---
        cv::Mat cpu_img = image.getAsCPU();
        cv::Mat gray_img;
        
        if (cpu_img.channels() == 3) {
            cv::cvtColor(cpu_img, gray_img, cv::COLOR_BGR2GRAY);
        } else {
            gray_img = cpu_img;
        }

        if (config.bucketing_params.enabled) {
            cv::Mat cpu_descriptors;
            
            // Factory lambda allows FeatureUtils to instantiate local SIFT objects per thread
            auto builder = [this]() {
                return cv::SIFT::create(nfeatures, nOctaveLayers, contrastThreshold, edgeThreshold, sigma);
            };
            
            // Dispatches to your multithreaded grid chopper
            FeatureUtils::detectWithGridCPU(builder, gray_img, out_keypoints, cpu_descriptors, config.bucketing_params, config.num_threads);
            out_descriptors = DeviceBuffer(cpu_descriptors);
        } else {
            cv::Mat cpu_descriptors;
            cv::setNumThreads(config.num_threads); // Maximize standard OpenCV CPU threading
            sift->detectAndCompute(gray_img, cv::noArray(), out_keypoints, cpu_descriptors);
            out_descriptors = DeviceBuffer(cpu_descriptors);
        }
    }
}
