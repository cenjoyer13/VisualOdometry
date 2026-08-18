#include "TrajectoryAligner.h"
#include <cmath>

void TrajectoryAligner::setYaw(double yaw) {
    const double c = std::cos(yaw), s = std::sin(yaw);
    T_ = (cv::Mat_<double>(4, 4) <<
          c, -s, 0, 0,
          s,  c, 0, 0,
          0,  0, 1, 0,
          0,  0, 0, 1);
    yaw_deg_ = std::remainder(yaw * 180.0 / M_PI, 360.0);
    ready_ = true;
}

bool TrajectoryAligner::update(const cv::Point2d& est_xy, const cv::Point2d& gt_xy,
                               double heading_rad, double dt) {
    if (ready_ || p_.mode == Mode::None) return false;

    if (p_.mode == Mode::Heading) {
        // Nothing to accumulate: the heading is already the answer.
        setYaw(p_.mount_offset_deg * M_PI / 180.0 - heading_rad);
        return true;
    }

    // --- GtDisplacement ---
    // Hold the start until the vehicle is actually translating. Below the gate
    // the displacement direction is wander, not heading, and re-anchoring keeps
    // the eventual baseline inside the moving segment instead of spanning the
    // hover that preceded it.
    const double step = cv::norm(gt_xy - gt_prev_);
    const double speed = (dt > 1e-6) ? step / dt : 0.0;
    gt_prev_ = gt_xy;

    if (!started_ || speed < p_.min_speed) {
        est_start_ = est_xy;
        gt_start_ = gt_xy;
        started_ = true;
        return false;      // min_speed 0 makes this the first call only
    }

    const cv::Point2d gt_disp = gt_xy - gt_start_;
    travelled_ = cv::norm(gt_disp);
    if (travelled_ < p_.init_distance) return false;

    const cv::Point2d est_disp = est_xy - est_start_;
    if (cv::norm(est_disp) < 1e-6) return false;

    setYaw(std::atan2(gt_disp.y, gt_disp.x) - std::atan2(est_disp.y, est_disp.x));
    return true;
}
