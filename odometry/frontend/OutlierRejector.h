#pragma once
#include <string>
#include <vector>
#include <opencv2/core.hpp>

// Two-view geometric outlier rejection, applied once for every frontend.
//
// WHY IT LIVES HERE AND NOT IN THE FRONTENDS
// ------------------------------------------
// Every frontend produces the same thing -- correspondence pairs between two
// views -- so rejection is a filter on that output, not a property of how the
// correspondences were found. Optical flow and descriptor matching would
// otherwise carry two copies of the same code.
//
// INPUTS ARE BEARINGS, NOT PIXELS. THIS IS NOT OPTIONAL.
// -----------------------------------------------------
// The epipolar constraint assumes a perspective camera. On a raw fisheye frame
// it simply does not hold, so running any of these models on raw pixels is the
// wrong model and would reject good matches hardest at the periphery -- exactly
// where UndistortMode::Points is trying to gain. Because OdometryPipeline owns
// pixel -> bearing for both undistort modes, one implementation here is correct
// for all four combinations of {image, points} x {optical flow, descriptors}.
//
// CHOOSING A MODEL
// ----------------
//   essential   5 dof (rotation + translation direction). The right choice with
//               a calibrated camera: the fewest degrees of freedom, so the
//               least slack for an outlier to hide in.
//   fundamental 7 dof. The uncalibrated form, related by E = K'^T F K. Two
//               extra dof that no real camera motion needs, so it keeps more
//               outliers. Kept because it makes no assumption about the lens
//               model, which makes it a diagnostic: if points-mode features are
//               being rejected because the Kannala-Brandt coefficients are poor
//               at the rim, essential and fundamental will disagree there.
//   homography  Plane-induced. Not a general-motion model, but the correct one
//               for a nadir camera over flat ground -- and the degenerate case
//               for BOTH of the above, since on a plane any F = [e']_x H fits,
//               leaving E and F under-determined by a two-parameter family.
//
// The threshold is quoted in pixels at kVirtualFocal so it stays comparable
// across cameras and matches VINS's F_threshold convention; internally it is
// divided by that focal to become a normalised-plane distance.
struct OutlierRejectionParams {
    std::string method = "none";   // none | essential | fundamental | homography
    std::string usac   = "magsac"; // magsac | accurate | ransac
    double threshold_px = 1.0;     // at kVirtualFocal
    double confidence   = 0.99;
    int    min_points   = 12;      // below this, skip rather than guess
};

class OutlierRejector {
public:
    // Pixels-per-radian the threshold is quoted against. 460 is VINS's own
    // virtual focal, kept so F_threshold values transfer directly.
    static constexpr double kVirtualFocal = 460.0;

    struct Result {
        bool ran = false;          // false when skipped; mask is then all-inliers
        int n_in = 0;
        int n_out = 0;
        const char* skip_reason = "";
    };

    explicit OutlierRejector(const OutlierRejectionParams& p);

    bool enabled() const { return method_ != 0; }
    const OutlierRejectionParams& params() const { return p_; }

    // `prev` and `curr` are index-aligned NORMALISED bearings. `mask` is sized
    // to match and set to 1 for inliers. On any skip path the mask is all 1s,
    // so callers never lose data because rejection could not run -- a model fit
    // on too few points, or a degenerate one, would otherwise throw away good
    // features rather than merely failing to remove bad ones.
    Result run(const std::vector<cv::Point2d>& prev,
               const std::vector<cv::Point2d>& curr,
               std::vector<unsigned char>& mask) const;

private:
    OutlierRejectionParams p_;
    int method_ = 0;      // 0 = none, else an internal model id
    int usac_flag_ = 0;   // cv::USAC_MAGSAC etc.
};
