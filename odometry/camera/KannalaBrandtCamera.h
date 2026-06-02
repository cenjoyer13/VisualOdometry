#pragma once
#include "ICameraModel.h"
#include "CameraModelConfig.h"

// Kannala-Brandt (equidistant fisheye) camera. Precomputes a remap to a
// rectified pinhole image at scale_factor x the calibrated resolution, using
// the calibrated mu/mv/u0/v0 (scaled) as the target pinhole matrix. The four
// config coefficients k2/k3/k4/k5 are OpenCV's fisheye D = [k2,k3,k4,k5].
class KannalaBrandtCamera : public ICameraModel {
public:
    explicit KannalaBrandtCamera(const CameraModelConfig& cfg);

    CameraIntrinsics intrinsics() const override { return intrinsics_; }
    cv::Size outputSize() const override { return output_size_; }
    cv::Mat undistortImage(const cv::Mat& raw) const override;

private:
    CameraIntrinsics intrinsics_;
    cv::Size output_size_;
    cv::Mat map1_, map2_;   // cv::remap lookup tables (CV_16SC2 + CV_16UC1)
};
