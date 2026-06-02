#include "PinholeCamera.h"
#include <opencv2/imgproc.hpp>

PinholeCamera::PinholeCamera(const CameraModelConfig& cfg)
    : scale_factor_(cfg.scale_factor > 0.0 ? cfg.scale_factor : 1.0) {
    const double s = scale_factor_;
    intrinsics_.fx = static_cast<float>(cfg.mu * s);
    intrinsics_.fy = static_cast<float>(cfg.mv * s);
    intrinsics_.cx = static_cast<float>(cfg.u0 * s);
    intrinsics_.cy = static_cast<float>(cfg.v0 * s);
    output_size_ = cv::Size(static_cast<int>(std::lround(cfg.image_width  * s)),
                            static_cast<int>(std::lround(cfg.image_height * s)));
}

cv::Mat PinholeCamera::undistortImage(const cv::Mat& raw) const {
    if (scale_factor_ == 1.0) return raw.clone();
    cv::Mat out;
    cv::resize(raw, out, output_size_, 0, 0, cv::INTER_AREA);
    return out;
}
