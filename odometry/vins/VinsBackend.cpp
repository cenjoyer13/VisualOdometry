#include "VinsBackend.h"

#include <iostream>
#include <map>
#include <utility>
#include <chrono>

#include "../../vins/vins_estimator/estimator/estimator.h"
#include "../../vins/vins_estimator/estimator/parameters.h"
#include "../utils/RunLog.h"

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
    // `freq` is vestigial in VINS-Fusion (VINS-Mono's FREQ throttle was replaced
    // by an every-other-frame drop tied to multiple_thread, which inputFeature()
    // bypasses), so read it here and enforce it ourselves. 0 disables.
    {
        cv::FileStorage fs(vins_config, cv::FileStorage::READ);
        if (fs.isOpened() && !fs["freq"].empty()) feed_hz_ = (double)fs["freq"];
    }
    // The backend's own resolved settings, as a run-scoped record. These are
    // the knobs that silently change results: SOLVER_TIME is a WALL-CLOCK
    // budget, so a small value makes the estimator nondeterministic; MAX_CNT is
    // the feature density the whole backend is tuned around; and TD /
    // ESTIMATE_TD decide whether the zero feature velocities we send are exact
    // or merely tolerated.
    LogRec("run.vins", /*stamp_frame=*/false)
        ("max_solver_time", SOLVER_TIME)
        ("max_num_iterations", NUM_ITERATIONS)
        ("max_cnt", MAX_CNT)
        ("min_parallax", MIN_PARALLAX)
        ("td", TD)
        ("estimate_td", ESTIMATE_TD)
        ("num_of_cam", NUM_OF_CAM)
        ("multiple_thread", MULTIPLE_THREAD)
        ("feed_hz", feed_hz_);

    std::cout << "[VinsBackend] VINS estimator ready (" << vins_config
              << "), feed throttle " << feed_hz_ << " Hz\n";
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
    // Feed-rate throttle, standing in for the one we bypassed. VINS's own
    // tracker only emits a featureFrame when it is below FREQ Hz (feature_tracker
    // .cpp's PUB_THIS_FRAME gate), and upstream additionally drops every other
    // frame when multiple_thread is 1. Going through inputFeature() skips both,
    // so without this the backend sees the raw image rate.
    //
    // That is not merely wasteful. The sliding window is a fixed 11 states, so
    // doubling the feed halves the time span it covers, and relativePose() then
    // cannot find two frames far enough apart to hit its 30-px average-parallax
    // threshold -- reported as "Not enough features or parallax", which is
    // exactly what this bag produced at the full ~16 Hz rate.
    if (feed_hz_ > 0.0) {
        const double min_dt = 1.0 / feed_hz_;
        if (last_fed_t_ > 0.0 && (t - last_fed_t_) < min_dt * 0.99) {
            LogRec("feed")("accepted", false)("n_feat", (int)ids.size());
            return;
        }
        last_fed_t_ = t;
    }

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

    // Feed record. VINS's initialiser fails on two distinct conditions
    // ("Not enough features or parallax" from relativePose, "IMU excitation not
    // enouth" from initialStructure) and telling them apart needs to know what
    // the frontend is actually delivering: the rate we feed at, how many
    // features per frame, and -- the one that matters most -- how many track
    // ids survive between consecutive frames. Short tracks starve
    // getCorresponding() of shared observations across the window no matter how
    // many features each individual frame carries.
    //
    // All three of the feed bugs found so far (10x feature count, 2x feed rate,
    // and the tracks themselves) were diagnosed from exactly these numbers.
    size_t carried = 0;
    for (size_t i = 0; i < n; ++i)
        if (prev_ids_.count(ids[i])) ++carried;
    LogRec("feed")
        ("accepted", true)
        ("dt", (last_logged_t_ > 0.0) ? (t - last_logged_t_) : 0.0)
        ("n_feat", (int)n)
        ("n_carried", (int)carried)
        ("carry", n ? double(carried) / double(n) : 0.0);
    last_logged_t_ = t;

    prev_ids_.clear();
    for (size_t i = 0; i < n; ++i) prev_ids_.insert(ids[i]);

    const auto t0 = std::chrono::steady_clock::now();
    est_->inputFeature(t, featureFrame);
    const double solve_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t0).count();

    logState(solve_ms);
}

void VinsBackend::logState(double solve_ms) {
    if (!RunLog::instance().enabled()) return;

    const bool init = isInitialized();
    LogRec r("state");
    r("solver_flag", init ? "NON_LINEAR" : "INITIAL")
     ("solve_ms", solve_ms)
     ("frame_count", est_->frame_count)
     ("marg", est_->marginalization_flag == Estimator::MarginalizationFlag::MARGIN_OLD
                  ? "OLD" : "NEW");
    if (!init) return;

    const int i = est_->frame_count;
    const double p[3]  = {est_->Ps[i](0),  est_->Ps[i](1),  est_->Ps[i](2)};
    const double v[3]  = {est_->Vs[i](0),  est_->Vs[i](1),  est_->Vs[i](2)};
    const double ba[3] = {est_->Bas[i](0), est_->Bas[i](1), est_->Bas[i](2)};
    const double bg[3] = {est_->Bgs[i](0), est_->Bgs[i](1), est_->Bgs[i](2)};
    const double g[3]  = {est_->g(0), est_->g(1), est_->g(2)};
    const Eigen::Quaterniond q(est_->Rs[i]);
    const double qv[4] = {q.x(), q.y(), q.z(), q.w()};

    r.vec("p", p, 3).vec("v", v, 3).vec("q", qv, 4)
     .vec("ba", ba, 3).vec("bg", bg, 3).vec("g", g, 3)
     ("v_norm", est_->Vs[i].norm())
     ("ba_norm", est_->Bas[i].norm())
     ("bg_norm", est_->Bgs[i].norm());
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
