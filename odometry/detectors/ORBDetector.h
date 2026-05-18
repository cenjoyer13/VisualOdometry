#pragma once
#include "IFeatureDetector.h"
#include "../OdometryTypes.h"
#include <opencv2/features2d.hpp>

// Forward declare CUDA ORB so we don't force the inclusion of CUDA 
// headers into the rest of the project pipeline.
namespace cv { namespace cuda { class ORB; } }

class ORBDetector : public IFeatureDetector {
private:
    OdometryConfig config;
    
    // CPU and OpenCL share the standard cv::ORB pointer
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
