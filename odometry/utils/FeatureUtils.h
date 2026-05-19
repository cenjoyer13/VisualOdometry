#pragma once
#include <vector>
#include <functional>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include "../OdometryTypes.h"

class FeatureUtils {
public:
    static void detectWithGridCPU(const std::function<cv::Ptr<cv::Feature2D>()>& detector_builder,
                                  const cv::Mat& image,
                                  std::vector<cv::KeyPoint>& out_keypoints,
                                  cv::Mat& out_descriptors,
                                  const BucketingConfig& bucketing_config,
                                  int num_threads);
};
