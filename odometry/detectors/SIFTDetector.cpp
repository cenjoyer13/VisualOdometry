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

    sift_cpu = cv::SIFT::create(nfeatures, nOctaveLayers, contrastThreshold, edgeThreshold, sigma);
}

void SIFTDetector::detect(DeviceBuffer& image, std::vector<cv::KeyPoint>& out_keypoints, DeviceBuffer& out_descriptors) {
    cv::Mat cpu_img = image.getAsCPU();
    cv::Mat gray_img;
    if (cpu_img.channels() == 3) cv::cvtColor(cpu_img, gray_img, cv::COLOR_BGR2GRAY);
    else gray_img = cpu_img;

    if (config.bucketing_params.enabled) {
        cv::Mat cpu_descriptors;
        
        // --- NEW: Lambda Factory for Multithreading ---
        auto builder = [this]() {
            return cv::SIFT::create(nfeatures, nOctaveLayers, contrastThreshold, edgeThreshold, sigma);
        };
        
        FeatureUtils::detectWithGridCPU(builder, gray_img, out_keypoints, cpu_descriptors, config.bucketing_params, config.num_threads);
        out_descriptors = DeviceBuffer(cpu_descriptors);
    } else {
        cv::Mat cpu_descriptors;
        sift_cpu->detectAndCompute(gray_img, cv::noArray(), out_keypoints, cpu_descriptors);
        out_descriptors = DeviceBuffer(cpu_descriptors);
    }
}
