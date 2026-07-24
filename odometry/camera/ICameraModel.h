#pragma once
#include <opencv2/core.hpp>
#include "../OdometryTypes.h"

// A camera model owns the mapping from a raw sensor image to the rectified
// pinhole image the pipeline consumes, plus the pinhole intrinsics that
// describe that rectified image. The pipeline core stays camera-model
// agnostic: undistortion is a preprocessing step the evaluator applies before
// processFrame, exactly like image decoding.
class ICameraModel {
public:
    virtual ~ICameraModel() = default;

    // Rectified pinhole intrinsics matching the output of undistortImage.
    // Seed OdometryConfig::intrinsics with this before building the pipeline.
    virtual CameraIntrinsics intrinsics() const = 0;

    // Size of the rectified image produced by undistortImage.
    virtual cv::Size outputSize() const = 0;

    // Map a raw sensor frame to the rectified pinhole frame. Returns a
    // caller-owned Mat of the same type as the input.
    virtual cv::Mat undistortImage(const cv::Mat& raw) const = 0;
};
