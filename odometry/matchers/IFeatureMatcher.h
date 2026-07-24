#pragma once
#include <vector>
#include <opencv2/core.hpp>
#include "../OdometryTypes.h"

class IFeatureMatcher {
public:
    virtual ~IFeatureMatcher() = default;

    // Descriptors arrive as DeviceBuffer; concrete matcher decides where to run.
    virtual std::vector<cv::DMatch> match(DeviceBuffer& desc_old,
                                          DeviceBuffer& desc_new,
                                          const std::vector<cv::KeyPoint>& kp_old,
                                          const std::vector<cv::KeyPoint>& kp_new) = 0;
};
