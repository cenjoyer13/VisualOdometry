#include "PinholeCamera.h"
#include <opencv2/imgproc.hpp>

PinholeCamera::PinholeCamera(const CameraModelConfig& cfg)
    : scale_factor_(cfg.scale_factor > 0.0 ? cfg.scale_factor : 1.0) {
    const double s = scale_factor_;
    mu_ = cfg.mu; mv_ = cfg.mv; u0_ = cfg.u0; v0_ = cfg.v0;
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

void PinholeCamera::liftProjective(const std::vector<cv::Point2f>& px,
                                   std::vector<cv::Point2d>& out) const {
    // This model carries no distortion coefficients (CameraModelConfig's k2..k5
    // are Kannala-Brandt only), so lifting is the plain pinhole inverse against
    // the RAW-pixel intrinsics -- not the scaled ones, because in Points mode
    // the frontend never saw a resized image.
    out.resize(px.size());
    for (size_t i = 0; i < px.size(); ++i) {
        out[i].x = (static_cast<double>(px[i].x) - u0_) / mu_;
        out[i].y = (static_cast<double>(px[i].y) - v0_) / mv_;
    }
}
