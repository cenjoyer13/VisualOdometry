#include "DualPathIntegrator.h"
#include <cmath>
#include <iostream>

DualPathIntegrator::DualPathIntegrator(const OdometryConfig& cfg) : config(cfg) {
    T_VO = cv::Mat::eye(4, 4, CV_64F);
    T_VIO = cv::Mat::eye(4, 4, CV_64F);
}

cv::Mat DualPathIntegrator::eulerToRotationMatrix(const cv::Vec3f& euler) const {
    double pitch = euler[0], roll = euler[1], yaw = euler[2];

    cv::Mat R_x = (cv::Mat_<double>(3, 3) <<
        1, 0, 0,
        0, std::cos(pitch), -std::sin(pitch),
        0, std::sin(pitch), std::cos(pitch));

    cv::Mat R_y = (cv::Mat_<double>(3, 3) <<
        std::cos(roll), 0, std::sin(roll),
        0, 1, 0,
        -std::sin(roll), 0, std::cos(roll));

    cv::Mat R_z = (cv::Mat_<double>(3, 3) <<
        std::cos(yaw), -std::sin(yaw), 0,
        std::sin(yaw), std::cos(yaw), 0,
        0, 0, 1);

    return R_z * R_y * R_x;
}

void DualPathIntegrator::snapshotPose(uint64_t frame_id) {
    snapshots_[frame_id] = T_VO.clone();
    snapshot_order_.push_back(frame_id);
    while (snapshot_order_.size() > kMaxSnapshots) {
        uint64_t evict = snapshot_order_.front();
        snapshot_order_.pop_front();
        snapshots_.erase(evict);
    }
}

bool DualPathIntegrator::getSnapshot(uint64_t frame_id, cv::Mat& out_T_world) const {
    auto it = snapshots_.find(frame_id);
    if (it == snapshots_.end()) return false;
    out_T_world = it->second.clone();
    return true;
}

bool DualPathIntegrator::applyCorrection(uint64_t frame_id, const cv::Mat& T_world_optimized) {
    auto it = snapshots_.find(frame_id);
    if (it == snapshots_.end()) {
        // Snapshot evicted; correction is no longer anchorable. Drop it.
        return false;
    }

    cv::Mat T_snap = it->second;
    cv::Mat T_opt;
    T_world_optimized.convertTo(T_opt, CV_64F);

    // Delta in world frame: applying delta to the snapshot recovers T_opt.
    //   T_opt = delta * T_snap   =>   delta = T_opt * T_snap.inv()
    cv::Mat delta = T_opt * T_snap.inv();

    // VO path: left-multiply so any future motion accumulated since the
    // snapshot remains correct relative to the corrected pose.
    T_VO = delta * T_VO;

    // VIO path: rotation is IMU-locked, so we only apply the translational
    // component of the world-frame delta. Rotation untouched.
    cv::Mat delta_t = delta(cv::Rect(3, 0, 1, 3));
    cv::Mat cur_t = T_VIO(cv::Rect(3, 0, 1, 3));
    cv::Mat new_t = cur_t + delta_t;
    new_t.copyTo(T_VIO(cv::Rect(3, 0, 1, 3)));

    return true;
}

void DualPathIntegrator::integrate(const cv::Mat& local_R, const cv::Mat& local_t, double scale, const cv::Vec3f& imu_orientation) {
    cv::Mat R_64, t_64;
    local_R.convertTo(R_64, CV_64F);
    local_t.convertTo(t_64, CV_64F);

    // PoseEstimator returns (R,t) such that x_curr = R*x_prev + t (i.e. T_curr_prev).
    // To track the camera in world frame we want T_prev_curr = inverse.
    cv::Mat R_cam = R_64.t();
    cv::Mat t_cam = -R_cam * t_64;

    cv::Mat scaled_t = t_cam * scale;

    cv::Mat T_local = cv::Mat::eye(4, 4, CV_64F);
    R_cam.copyTo(T_local(cv::Rect(0, 0, 3, 3)));
    scaled_t.copyTo(T_local(cv::Rect(3, 0, 1, 3)));

    // VO: T_global = T_global * T_local
    T_VO = T_VO * T_local;

    // VIO: rotation from IMU absolute, translation accumulated using IMU rotation
    cv::Mat R_imu = eulerToRotationMatrix(imu_orientation);
    cv::Mat current_VIO_t = T_VIO(cv::Rect(3, 0, 1, 3)).clone();
    cv::Mat step_global = R_imu * scaled_t;
    cv::Mat new_VIO_t = current_VIO_t + step_global;

    R_imu.copyTo(T_VIO(cv::Rect(0, 0, 3, 3)));
    new_VIO_t.copyTo(T_VIO(cv::Rect(3, 0, 1, 3)));
}

cv::Mat DualPathIntegrator::getGlobalTransformVO() const { return T_VO.clone(); }
cv::Mat DualPathIntegrator::getGlobalTransformVIO() const { return T_VIO.clone(); }
