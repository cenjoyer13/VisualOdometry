#pragma once
#include "IFeatureDetector.h"
#include "../OdometryTypes.h"
#include <opencv2/features2d.hpp>

// Forward declared: keeps CUDA headers out of files that include this one.
namespace cv { namespace cuda { class ORB; } }

class ORBDetector : public IFeatureDetector {
private:
    OdometryConfig config;
    
    // CPU and OpenCL paths share this; OpenCL is selected via cv::UMat input.
    cv::Ptr<cv::ORB> orb_cpu;

#ifdef HAS_CUDA
    cv::Ptr<cv::cuda::ORB> orb_cuda;
#endif

public:
    explicit ORBDetector(const OdometryConfig& cfg);
    ~ORBDetector() override = default;

    void detect(DeviceBuffer& image, 
                std::vector<cv::KeyPoint>& out_keypoints, 
                DeviceBuffer& out_descriptors) override;
};
