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
    
    static void filterByGrid(const std::vector<cv::KeyPoint>& in_kpts,
    		      const cv::Mat& in_desc,
    		      std::vector<cv::KeyPoint>& out_kpts,
    		      cv::Mat& out_desc,
    		      int image_width, int image_height,
    		      const BucketingConfig& params);
};
