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

    // Resolved cv::ORB::create() args, kept as members so the per-thread
    // bucketing builder (detectWithGridCPU) can construct its own ORB
    // instance per worker with the same parameters (cv::ORB is not
    // thread-safe, same reasoning as SIFTDetector's builder).
    int num_features;
    float scale_factor;
    int nlevels;
    int edge_threshold;
    int first_level;
    int wta_k;
    int patch_size;
    int fast_threshold;

public:
    explicit ORBDetector(const OdometryConfig& cfg);
    ~ORBDetector() override = default;

    void detect(DeviceBuffer& image, 
                std::vector<cv::KeyPoint>& out_keypoints, 
                DeviceBuffer& out_descriptors) override;
};
