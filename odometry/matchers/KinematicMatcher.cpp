#include "KinematicMatcher.h"
#include <iostream>
#include <limits>

#ifdef HAS_CUDA
#include <opencv2/cudafeatures2d.hpp>
#endif

KinematicMatcher::KinematicMatcher(const OdometryConfig& cfg) : config(cfg) {
    // Matcher creation is deferred until match(). Norm type depends on the
    // descriptor data type (binary for ORB, float for SIFT/SuperPoint), which
    // is only known once the detector emits its first batch.
}

void KinematicMatcher::ensureInitialized(int desc_depth) {
    if (matcher_cpu) return;

    norm_type = (desc_depth == CV_8U) ? cv::NORM_HAMMING : cv::NORM_L2;

    matcher_cpu = cv::BFMatcher::create(norm_type);

#ifdef HAS_CUDA
    if (config.backend == ComputeBackend::CUDA) {
        matcher_cuda = cv::cuda::DescriptorMatcher::createBFMatcher(norm_type);
    }
#endif

    if (config.verbose) {
        std::cout << "[KinematicMatcher] Norm: "
                  << (norm_type == cv::NORM_HAMMING ? "HAMMING (binary)" : "L2 (float)")
                  << std::endl;
    }
}

std::vector<cv::DMatch> KinematicMatcher::match(DeviceBuffer& desc_old,
                                                DeviceBuffer& desc_new,
                                                const std::vector<cv::KeyPoint>& kp_old,
                                                const std::vector<cv::KeyPoint>& kp_new) {

    // Empty input: nothing to match.
    if (kp_old.empty() || kp_new.empty()) {
        return {};
    }

    // Descriptor-type probe: HAMMING for binary (CV_8U), L2 for float.
    cv::Mat probe = desc_old.getAsCPU();
    ensureInitialized(probe.depth());

    // Tunables with literal fallbacks.
    float ratio_thresh = config.matcher_params.count("ratio_thresh") ? config.matcher_params.at("ratio_thresh") : 0.75f;
    float max_pixel_jump = config.matcher_params.count("max_pixel_jump") ? config.matcher_params.at("max_pixel_jump") : 80.0f;

    // Absolute distance cap. Accepts the legacy `max_hamming` key for binary
    // configs; otherwise reads `max_distance`. Default for HAMMING is ~50
    // bits. For L2 the cap is effectively disabled: OpenCV SIFT emits
    // unnormalised descriptors (good-match distances ~100-300) while
    // SuperPoint emits unit-normalised ones (~0-1.4), so no single literal
    // works for both. The ratio test does the real filtering; an L2 cap can
    // still be set explicitly via `max_distance`.
    float max_distance;
    if (config.matcher_params.count("max_distance")) {
        max_distance = config.matcher_params.at("max_distance");
    } else if (config.matcher_params.count("max_hamming")) {
        max_distance = config.matcher_params.at("max_hamming");
    } else {
        max_distance = (norm_type == cv::NORM_HAMMING) ? 50.0f : std::numeric_limits<float>::max();
    }

    // knn with k=2: required by the ratio test below.
    std::vector<std::vector<cv::DMatch>> knn_matches;

    // Hardware-routed knnMatch.
    switch (config.backend) {
        case ComputeBackend::CUDA: {
#ifdef HAS_CUDA
            cv::cuda::GpuMat* d_old = static_cast<cv::cuda::GpuMat*>(desc_old.getCUDAPointer());
            cv::cuda::GpuMat* d_new = static_cast<cv::cuda::GpuMat*>(desc_new.getCUDAPointer());
            matcher_cuda->knnMatch(*d_old, *d_new, knn_matches, 2);
            break;
#else
            if (config.verbose) {
                static bool warned_cuda = false;
                if (!warned_cuda) {
                    std::cerr << "[KinematicMatcher] CUDA requested but not compiled. Falling back to CPU." << std::endl;
                    warned_cuda = true;
                }
            }
            // Intentional fallthrough: next case is OPENCL, which itself
            // cascades to CPU when OpenCL is unavailable at runtime.
#endif
        }
        case ComputeBackend::OPENCL: {
            // UMat input selects the OpenCL kernel path inside the T-API.
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

    // Filtering runs on the host; keypoints are small and the kept fraction
    // is much smaller than the input.
    std::vector<cv::DMatch> good_matches;
    good_matches.reserve(knn_matches.size());

    for (const auto& match_pair : knn_matches) {
        // ratio test needs both candidates.
        if (match_pair.size() < 2) continue;

        const cv::DMatch& best = match_pair[0];
        const cv::DMatch& second_best = match_pair[1];

        // Lowe's ratio test combined with the absolute-distance cap.
        if (best.distance < ratio_thresh * second_best.distance && best.distance < max_distance) {

            // Pixel-jump cap: rejects matches that move farther than is
            // physically plausible between adjacent frames.
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
