#pragma once
#include <vector>
#include <opencv2/core.hpp>
#include "OdometryTypes.h"

class IFeatureDetector {
public:
    virtual ~IFeatureDetector() = default;

    // Takes a device-agnostic image, outputs device-agnostic descriptors.
    // Keypoints are kept as std::vector since they are tiny (few KB) and 
    // usually needed on the CPU for pose estimation math anyway.
    virtual void detect(DeviceBuffer& image, 
                        std::vector<cv::KeyPoint>& out_keypoints, 
                        DeviceBuffer& out_descriptors) = 0;
};
