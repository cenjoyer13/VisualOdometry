#pragma once
#include "IFeatureMatcher.h"
#include "../OdometryTypes.h"
#include <onnxruntime_cxx_api.h>
#include <memory>
#include <vector>

class LightGlueMatcher : public IFeatureMatcher {
private:
    OdometryConfig config;
    
    // ONNX Runtime ecosystem
    std::unique_ptr<Ort::Env> env;
    std::unique_ptr<Ort::Session> session;
    Ort::MemoryInfo memory_info_cpu;

    bool is_sift;

    // Internal initialization and math helpers
    void initializeSession(int desc_dim);
    void convertToRootSift(cv::Mat& desc);

public:
    explicit LightGlueMatcher(const OdometryConfig& cfg);
    ~LightGlueMatcher() override = default;

    std::vector<cv::DMatch> match(DeviceBuffer& desc_old, 
                                  DeviceBuffer& desc_new,
                                  const std::vector<cv::KeyPoint>& kp_old,
                                  const std::vector<cv::KeyPoint>& kp_new) override;
};
