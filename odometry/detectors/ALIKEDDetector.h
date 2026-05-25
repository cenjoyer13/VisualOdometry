#pragma once

#include "IFeatureDetector.h"
#include "../OdometryTypes.h"
#include <onnxruntime_cxx_api.h>
#include <opencv2/core.hpp>
#include <memory>

class ALIKEDDetector : public IFeatureDetector {
private:
    OdometryConfig config;
    
    // ONNX Runtime components
    std::unique_ptr<Ort::Env> env;
    std::unique_ptr<Ort::Session> session;
    Ort::MemoryInfo memory_info_cpu;

    // ALIKED parameters with fallbacks
    int max_keypoints;
    float detection_threshold;
    int nms_radius;

    void initializeSession();
    void preprocess(const cv::Mat& image, std::vector<float>& input_tensor_values, std::vector<int64_t>& input_shape);

public:
    explicit ALIKEDDetector(const OdometryConfig& cfg);
    ~ALIKEDDetector() override = default;

    void detect(DeviceBuffer& image, std::vector<cv::KeyPoint>& out_keypoints, DeviceBuffer& out_descriptors) override;
};
