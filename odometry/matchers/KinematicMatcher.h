#pragma once
#include "IFeatureMatcher.h"
#include "../OdometryTypes.h"
#include <opencv2/features2d.hpp>

// Forward declared: keeps CUDA headers out of files that include this one.
namespace cv { namespace cuda { class DescriptorMatcher; } }

class KinematicMatcher : public IFeatureMatcher {
private:
    OdometryConfig config;

    // Norm: resolved on first descriptor batch. NORM_HAMMING for ORB (CV_8U),
    // NORM_L2 for SIFT / SuperPoint (CV_32F). -1 marks "not yet initialized".
    int norm_type = -1;

    // BF matcher used for both CPU and OpenCL paths; OpenCL is selected when
    // the input descriptors arrive as cv::UMat.
    cv::Ptr<cv::DescriptorMatcher> matcher_cpu;

#ifdef HAS_CUDA
    cv::Ptr<cv::cuda::DescriptorMatcher> matcher_cuda;
#endif

    void ensureInitialized(int desc_depth);

public:
    explicit KinematicMatcher(const OdometryConfig& cfg);
    ~KinematicMatcher() override = default;

    std::vector<cv::DMatch> match(DeviceBuffer& desc_old,
                                  DeviceBuffer& desc_new,
                                  const std::vector<cv::KeyPoint>& kp_old,
                                  const std::vector<cv::KeyPoint>& kp_new) override;
};
