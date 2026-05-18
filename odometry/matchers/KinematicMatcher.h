#pragma once
#include "IFeatureMatcher.h"
#include "../OdometryTypes.h"
#include <opencv2/features2d.hpp>

// Forward declare to avoid CUDA header bleeding
namespace cv { namespace cuda { class DescriptorMatcher; } }

class KinematicMatcher : public IFeatureMatcher {
private:
    OdometryConfig config;
    
    // Core OpenCV Brute-Force Matcher (handles both CPU and OpenCL automatically)
    cv::Ptr<cv::DescriptorMatcher> matcher_cpu;

#ifdef HAS_CUDA
    cv::Ptr<cv::cuda::DescriptorMatcher> matcher_cuda;
#endif

public:
    explicit KinematicMatcher(const OdometryConfig& cfg);
    ~KinematicMatcher() override = default;

    std::vector<cv::DMatch> match(DeviceBuffer& desc_old, 
                                  DeviceBuffer& desc_new,
                                  const std::vector<cv::KeyPoint>& kp_old,
                                  const std::vector<cv::KeyPoint>& kp_new) override;
};
