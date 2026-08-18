#include "VinsBackend.h"

#include <iostream>
#include <map>
#include <utility>

#include "../../vins/vins_estimator/estimator/estimator.h"
#include "../../vins/vins_estimator/estimator/parameters.h"

VinsBackend::VinsBackend(const std::string& vins_config) {
    // Order matters and is inherited from rosNodeTest.cpp: readParameters()
    // fills the file-scope globals (RIC/TIC, noise densities, TD, WINDOW size
    // derived constants, CAM_NAMES) that both the Estimator constructor and
    // setParameter() read. Constructing the Estimator first would capture
    // defaults.
    readParameters(const_cast<std::string&>(vins_config));
    est_ = std::make_unique<Estimator>();
    est_->setParameter();

    if (MULTIPLE_THREAD) {
        // Not fatal, but replay stops being reproducible: inputImage() drops
        // every other frame and processMeasurements() races the feed loop.
        std::cout << "[VinsBackend] WARNING: multiple_thread is 1 in "
                  << vins_config << ". Set it to 0 for deterministic replay.\n";
    }
    std::cout << "[VinsBackend] VINS estimator ready (" << vins_config << ")\n";
}

VinsBackend::~VinsBackend() = default;

void VinsBackend::addImu(double t, const cv::Vec3d& acc, const cv::Vec3d& gyr) {
    est_->inputIMU(t,
                   Eigen::Vector3d(acc[0], acc[1], acc[2]),
                   Eigen::Vector3d(gyr[0], gyr[1], gyr[2]));
}

void VinsBackend::addFrame(double t,
                           const std::vector<int64_t>& ids,
                           const std::vector<cv::Point2f>& pts,
                           const CameraIntrinsics& K) {
    // featureFrame layout mirrors FeatureTracker::trackImage()'s return value
    // exactly -- see the header for why velocity is left at zero.
    std::map<int, std::vector<std::pair<int, Eigen::Matrix<double, 7, 1>>>> featureFrame;

    const size_t n = std::min(ids.size(), pts.size());
    for (size_t i = 0; i < n; ++i) {
        const double u = pts[i].x;
        const double v = pts[i].y;
        const double x = (u - K.cx) / K.fx;   // rectified pinhole -> normalized
        const double y = (v - K.cy) / K.fy;

        Eigen::Matrix<double, 7, 1> xyz_uv_velocity;
        xyz_uv_velocity << x, y, 1.0, u, v, 0.0, 0.0;
        // camera_id 0: monocular. A second camera would append (1, ...) under
        // the SAME feature id, which is how VINS recognizes a stereo pair.
        featureFrame[static_cast<int>(ids[i])].emplace_back(0, xyz_uv_velocity);
    }

    est_->inputFeature(t, featureFrame);
}

bool VinsBackend::isInitialized() const {
    return est_->solver_flag == Estimator::SolverFlag::NON_LINEAR;
}

bool VinsBackend::latestPose(cv::Mat& T_world_body) const {
    if (!isInitialized()) return false;

    // frame_count indexes the newest state in the sliding window; once
    // initialized it sits pinned at WINDOW_SIZE.
    const int i = est_->frame_count;
    const Eigen::Matrix3d& R = est_->Rs[i];
    const Eigen::Vector3d& p = est_->Ps[i];

    T_world_body = cv::Mat::eye(4, 4, CV_64F);
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) T_world_body.at<double>(r, c) = R(r, c);
        T_world_body.at<double>(r, 3) = p(r);
    }
    return true;
}

bool VinsBackend::latestState(cv::Vec3d& v, cv::Vec3d& ba, cv::Vec3d& bg) const {
    if (!isInitialized()) return false;
    const int i = est_->frame_count;
    v  = cv::Vec3d(est_->Vs[i](0),  est_->Vs[i](1),  est_->Vs[i](2));
    ba = cv::Vec3d(est_->Bas[i](0), est_->Bas[i](1), est_->Bas[i](2));
    bg = cv::Vec3d(est_->Bgs[i](0), est_->Bgs[i](1), est_->Bgs[i](2));
    return true;
}
