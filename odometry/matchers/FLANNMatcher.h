#pragma once
#include "IFeatureMatcher.h"
#include "../OdometryTypes.h"
#include <opencv2/features2d.hpp>

class FLANNMatcher : public IFeatureMatcher {
private:
    OdometryConfig config;
    cv::Ptr<cv::DescriptorMatcher> flann_matcher;

public:
    explicit FLANNMatcher(const OdometryConfig& cfg);
    ~FLANNMatcher() override = default;

    std::vector<cv::DMatch> match(DeviceBuffer& desc_old, 
                                  DeviceBuffer& desc_new,
                                  const std::vector<cv::KeyPoint>& kp_old,
                                  const std::vector<cv::KeyPoint>& kp_new) override;
};
