#include "FLANNMatcher.h"
#include <iostream>

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

    cv::Mat cpu_old = desc_old.getAsCPU();
    cv::Mat cpu_new = desc_new.getAsCPU();

    // SAFETY CHECK: FLANN KD-Tree requires Float32 descriptors (SIFT provides this natively).
    // If ORB (uint8) was accidentally passed, we convert it to avoid a core dump.
    if (cpu_old.type() != CV_32F) cpu_old.convertTo(cpu_old, CV_32F);
    if (cpu_new.type() != CV_32F) cpu_new.convertTo(cpu_new, CV_32F);

    std::vector<std::vector<cv::DMatch>> knn_matches;
    flann_matcher->knnMatch(cpu_old, cpu_new, knn_matches, 2);

    // Apply Lowe's Ratio Test
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
