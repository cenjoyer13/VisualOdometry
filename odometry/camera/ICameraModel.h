#pragma once
#include <vector>
#include <opencv2/core.hpp>
#include "../OdometryTypes.h"

// A camera model owns the mapping between raw sensor pixels and bearings.
//
// TWO WAYS TO REMOVE DISTORTION, AND WHY BOTH EXIST
// ------------------------------------------------
// UndistortMode::Image  -- rectify the whole frame to a pinhole image, run the
//   frontend on that, and normalise with the plain (u - cx) / fx. Simple, and
//   descriptors behave because the image is a true perspective projection. The
//   cost is severe: rectifying a fisheye to a pinhole keeps only the central
//   crop the target focal spans (with this nadir camera's fx = 414.6 over 720
//   px, about 82 degrees horizontally), so the periphery is discarded outright.
//   It also resamples every pixel, softening the detail features are found in.
//
// UndistortMode::Points -- run the frontend on the RAW frame and lift only the
//   feature pixels to bearings. Keeps the full field of view, resamples
//   nothing, and is what VINS itself does natively. Two caveats: the rim relies
//   on distortion coefficients that were fitted mostly from mid-radius
//   observations and extrapolate poorly, and descriptor matchers degrade there
//   because a patch's apparent shape changes with viewpoint under strong
//   distortion (optical flow, tracking small frame-to-frame displacements, is
//   far more tolerant).
//
// Neither is universally right, which is why the mode is a config key rather
// than a decision baked into the code: the effect on any given frontend is a
// thing to measure, not to argue about.
class ICameraModel {
public:
    virtual ~ICameraModel() = default;

    // Pinhole intrinsics of the image undistortImage() produces. Only
    // meaningful in Image mode; in Points mode the frontend sees raw pixels and
    // liftProjective() owns the geometry.
    virtual CameraIntrinsics intrinsics() const = 0;

    // Size of the image undistortImage() produces.
    virtual cv::Size outputSize() const = 0;

    // Raw sensor frame -> rectified pinhole frame.
    virtual cv::Mat undistortImage(const cv::Mat& raw) const = 0;

    // RAW sensor pixels -> normalised coordinates on the unit-depth plane
    // (x/z, y/z). This is the Points-mode counterpart of undistortImage: it
    // applies the same lens model to a sparse set of points instead of to every
    // pixel. `out` is resized to match `px`.
    //
    // Output is double, not float, and deliberately so. Pixels are float
    // because that is what the detectors produce, but the bearing feeds a
    // nonlinear solve: rounding it to float shifts the estimate by ~2e-9 m on
    // the first frame and 6.5 cm by the end of a 300 s replay. Measured, not
    // hypothetical -- it is what the image-mode regression caught.
    virtual void liftProjective(const std::vector<cv::Point2f>& px,
                                std::vector<cv::Point2d>& out) const = 0;
};
