#pragma once
#include "IFeatureDetector.h"
#include "../OdometryTypes.h"
#include <opencv2/features2d.hpp>
#include <memory>

class SIFTDetector : public IFeatureDetector {
private:
    OdometryConfig config;

    cv::Ptr<cv::SIFT> sift;

    int nfeatures;
    int nOctaveLayers;
    double contrastThreshold;
    double edgeThreshold;
    double sigma;

public:
    explicit SIFTDetector(const OdometryConfig& cfg);
    ~SIFTDetector() override = default;

    void detect(DeviceBuffer& image, 
                std::vector<cv::KeyPoint>& out_keypoints, 
                DeviceBuffer& out_descriptors) override;
};
