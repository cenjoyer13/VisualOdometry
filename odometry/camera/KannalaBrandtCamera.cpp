#include "KannalaBrandtCamera.h"
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

KannalaBrandtCamera::KannalaBrandtCamera(const CameraModelConfig& cfg) {
    const double s = cfg.scale_factor > 0.0 ? cfg.scale_factor : 1.0;

    // Calibrated fisheye intrinsics + the four KB polynomial coefficients.
    cv::Mat K = (cv::Mat_<double>(3, 3) <<
        cfg.mu, 0.0,    cfg.u0,
        0.0,    cfg.mv, cfg.v0,
        0.0,    0.0,    1.0);
    cv::Mat D = (cv::Mat_<double>(4, 1) << cfg.k2, cfg.k3, cfg.k4, cfg.k5);

    // Target pinhole matrix: same focal/principal point, scaled to the output
    // resolution. Keeping the calibrated focal yields a central, fixed crop of
    // the fisheye FOV rather than an FOV-fitting reprojection.
    output_size_ = cv::Size(static_cast<int>(std::lround(cfg.image_width  * s)),
                            static_cast<int>(std::lround(cfg.image_height * s)));
    cv::Mat newK = (cv::Mat_<double>(3, 3) <<
        cfg.mu * s, 0.0,        cfg.u0 * s,
        0.0,        cfg.mv * s, cfg.v0 * s,
        0.0,        0.0,        1.0);

    cv::fisheye::initUndistortRectifyMap(
        K, D, cv::Mat::eye(3, 3, CV_64F), newK, output_size_,
        CV_16SC2, map1_, map2_);

    intrinsics_.fx = static_cast<float>(cfg.mu * s);
    intrinsics_.fy = static_cast<float>(cfg.mv * s);
    intrinsics_.cx = static_cast<float>(cfg.u0 * s);
    intrinsics_.cy = static_cast<float>(cfg.v0 * s);
}

cv::Mat KannalaBrandtCamera::undistortImage(const cv::Mat& raw) const {
    cv::Mat out;
    cv::remap(raw, out, map1_, map2_, cv::INTER_LINEAR, cv::BORDER_CONSTANT);
    return out;
}
