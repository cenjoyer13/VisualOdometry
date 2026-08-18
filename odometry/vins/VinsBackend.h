#pragma once
#include <memory>
#include <string>
#include <vector>
#include <cstdint>

#include <opencv2/core.hpp>
#include <eigen3/Eigen/Dense>

#include "../OdometryTypes.h"

class Estimator;

// Adapter between this project's frontend and the vendored VINS-Fusion
// estimator (vins/vins_estimator).
//
// WHY AN ADAPTER AND NOT Estimator::inputImage()
// ----------------------------------------------
// inputImage() runs VINS's own goodFeaturesToTrack + KLT tracker and then
// pushes the result onto featureBuf. inputFeature() pushes onto that same
// buffer directly, so feeding our own correspondences is a first-class path
// rather than a hack -- and it is the whole point of this fork: the frontend
// (detector / matcher / optical flow) becomes the experimental variable while
// the backend stays a known-good VIO.
//
// WHAT THE BACKEND EXPECTS PER FRAME
// ----------------------------------
// A map feature_id -> [(camera_id, [x, y, 1, u, v, vx, vy])] where x/y are
// UNDISTORTED NORMALIZED coordinates (unit-depth plane), u/v are pixels, and
// vx/vy are normalized-plane velocities in 1/s. Two consequences:
//
//   * Feature IDs must be PERSISTENT ACROSS FRAMES. VINS builds per-feature
//     inverse-depth landmarks and decides marginalization from each feature's
//     parallax history (FeatureManager::addFeatureCheckParallax), so ids that
//     reset every frame leave it with no tracks and it will never initialize.
//     OpticalFlowFrontend already satisfies this -- its KLT track ids survive
//     promoteKeyframe(). A descriptor frontend does not, and needs a track
//     manager before it can be fed here.
//   * Velocities are only consumed by the td-compensation term of the
//     projection factors. With estimate_td: 0 the optimized td stays pinned at
//     the configured value and that term cancels, so zeros are exact rather
//     than merely tolerable. Revisit if estimate_td is ever turned on.
//
// Pixels arriving here are already RECTIFIED PINHOLE pixels: the evaluator
// applies ICameraModel::undistortImage() before the frontend runs, so
// normalization is the plain (u - cx) / fx. Do not undistort twice.
class VinsBackend {
public:
    // `vins_config` is a stock VINS-Fusion yaml (imu/cam topics, body_T_cam0,
    // noise densities, td). It is read by VINS's own readParameters(), so the
    // dataset's proven calibration transfers verbatim instead of being
    // re-derived from this project's schema. Relative paths inside it (notably
    // cam0_calib) resolve against the config file's own directory, matching
    // upstream behavior.
    //
    // No ROS master or ros::init() needed: the vendored Estimator's NodeHandle
    // was removed along with the lidar debug publishers.
    explicit VinsBackend(const std::string& vins_config);
    ~VinsBackend();

    // IMU sample on the IMU clock. Units: m/s^2 and rad/s.
    void addImu(double t, const cv::Vec3d& acc, const cv::Vec3d& gyr);

    // One frame of tracked features. `t` is the image timestamp on the IMU
    // clock (VINS applies its own td on top). `ids` and `pts` must be the same
    // length and index-aligned; `pts` are rectified pinhole pixels and `K` the
    // matching rectified intrinsics. Feeds every frame, not just keyframes --
    // VINS makes its own keyframe/marginalization decision from parallax.
    void addFrame(double t,
                  const std::vector<int64_t>& ids,
                  const std::vector<cv::Point2f>& pts,
                  const CameraIntrinsics& K);

    // True once VINS has finished initialization and is running the nonlinear
    // solver. Before that the reported pose is meaningless.
    bool isInitialized() const;

    // Latest optimized body pose as a 4x4 CV_64F world transform.
    // Returns false (T untouched) until isInitialized().
    bool latestPose(cv::Mat& T_world_body) const;

    // Latest optimized body velocity, biases and the window's gravity estimate.
    // Diagnostics; valid once isInitialized().
    bool latestState(cv::Vec3d& v, cv::Vec3d& ba, cv::Vec3d& bg) const;

private:
    std::unique_ptr<Estimator> est_;
};
