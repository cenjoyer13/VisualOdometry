#pragma once
#include <vector>
#include <opencv2/core.hpp>
#include "../OdometryTypes.h"

class IFeatureDetector {
public:
    virtual ~IFeatureDetector() = default;

    // I/O: DeviceBuffer for image and descriptors keeps the interface
    // backend-agnostic. Keypoints stay as a host std::vector; they are
    // small and pose estimation consumes them on the CPU.
    virtual void detect(DeviceBuffer& image, 
                        std::vector<cv::KeyPoint>& out_keypoints, 
                        DeviceBuffer& out_descriptors) = 0;
};
