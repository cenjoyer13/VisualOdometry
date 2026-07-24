#pragma once
#include "IFeatureDetector.h"
#include "../OdometryTypes.h"
#include <onnxruntime_cxx_api.h>
#include <memory>
#include <vector>

class SuperPointDetector : public IFeatureDetector {
private:
    OdometryConfig config;

    std::unique_ptr<Ort::Env> env;
    std::unique_ptr<Ort::Session> session;
    Ort::MemoryInfo memory_info_cpu;

    void initializeSession();

public:
    explicit SuperPointDetector(const OdometryConfig& cfg);
    ~SuperPointDetector() override = default;

    void detect(DeviceBuffer& image, 
                std::vector<cv::KeyPoint>& out_keypoints, 
                DeviceBuffer& out_descriptors) override;
};
