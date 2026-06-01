#define _USE_MATH_DEFINES  // Must be before any header that transitively pulls <cmath>.
#include "LocalBundleAdjustment.h"
#include <iostream>
#include <cmath>
#include <limits>
#include <algorithm>

#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Cal3_S2.h>
#include <gtsam/slam/SmartProjectionPoseFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/base/Vector.h>

using gtsam::symbol_shorthand::X;

LocalBundleAdjustment::LocalBundleAdjustment(const cv::Mat& K, const LBAParams& params, bool verbose)
    : params_(params),
      verbose_(verbose),
      K_(K.clone()),
      is_running_(false) {
    if (params_.opt_stride <= 0) params_.opt_stride = 1;
    if (params_.window_size <= 0) params_.window_size = 1;
}

LocalBundleAdjustment::~LocalBundleAdjustment() { stop(); }

void LocalBundleAdjustment::start() {
    if (!is_running_) {
        is_running_ = true;
        ba_thread_ = std::thread(&LocalBundleAdjustment::optimizationLoop, this);
    }
}

void LocalBundleAdjustment::stop() {
    if (is_running_) {
        is_running_ = false;
        cv_.notify_one();
        if (ba_thread_.joinable()) ba_thread_.join();
    }
}

void LocalBundleAdjustment::pushFrame(const BAFrame& frame) {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    pending_frames_.push(frame);
    cv_.notify_one();
}

void LocalBundleAdjustment::optimizationLoop() {
    while (is_running_) {
        BAFrame current_frame;
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            cv_.wait(lock, [this]() { return !pending_frames_.empty() || !is_running_; });
            if (!is_running_ && pending_frames_.empty()) break;

            current_frame = pending_frames_.front();
            pending_frames_.pop();
        }

        assignTrackIDs(current_frame);

        {
            std::lock_guard<std::mutex> lock(window_mutex_);
            local_window_.push_back(current_frame);
            frames_since_last_opt_++;

            if (local_window_.size() > static_cast<size_t>(params_.window_size)) {
                pruneOutdatedTracks(local_window_.front());
                local_window_.erase(local_window_.begin());
            }
        }

        if (local_window_.size() == static_cast<size_t>(params_.window_size) &&
            frames_since_last_opt_ >= params_.opt_stride) {
            runOptimization();
            frames_since_last_opt_ = 0;
        }
    }
}

void LocalBundleAdjustment::assignTrackIDs(BAFrame& frame) {
    frame.track_ids.assign(frame.points2D.size(), -1);

    for (size_t i = 0; i < frame.points2D.size(); ++i) {
        int prev_idx = frame.matched_prev_idx[i];

        if (prev_idx != -1 &&
            static_cast<size_t>(prev_idx) < prev_frame_track_ids_.size() &&
            prev_frame_track_ids_[prev_idx] != -1) {
            frame.track_ids[i] = prev_frame_track_ids_[prev_idx];
        } else {
            frame.track_ids[i] = next_track_id_++;
        }

        active_landmarks_[frame.track_ids[i]]++;
    }

    prev_frame_track_ids_ = frame.track_ids;
}

void LocalBundleAdjustment::pruneOutdatedTracks(const BAFrame& old_frame) {
    for (int64_t t_id : old_frame.track_ids) {
        if (t_id != -1) {
            active_landmarks_[t_id]--;
            if (active_landmarks_[t_id] <= 0) {
                active_landmarks_.erase(t_id);
            }
        }
    }
}

// Helpers ------------------------------------------------------------

static cv::Mat makeT(const cv::Mat& R, const cv::Mat& t) {
    cv::Mat T = cv::Mat::eye(4, 4, CV_64F);
    cv::Mat R64, t64;
    R.convertTo(R64, CV_64F);
    t.convertTo(t64, CV_64F);
    R64.copyTo(T(cv::Rect(0, 0, 3, 3)));
    t64.copyTo(T(cv::Rect(3, 0, 1, 3)));
    return T;
}

static gtsam::Pose3 cvMatToPose3(const cv::Mat& T_in) {
    cv::Mat T;
    T_in.convertTo(T, CV_64F);
    gtsam::Rot3 R(
        T.at<double>(0,0), T.at<double>(0,1), T.at<double>(0,2),
        T.at<double>(1,0), T.at<double>(1,1), T.at<double>(1,2),
        T.at<double>(2,0), T.at<double>(2,1), T.at<double>(2,2));
    gtsam::Point3 t(T.at<double>(0,3), T.at<double>(1,3), T.at<double>(2,3));
    return gtsam::Pose3(R, t);
}

static cv::Mat pose3ToCvMat(const gtsam::Pose3& p) {
    gtsam::Matrix4 m = p.matrix();
    cv::Mat T(4, 4, CV_64F);
    for (int r = 0; r < 4; r++)
        for (int c = 0; c < 4; c++)
            T.at<double>(r, c) = m(r, c);
    return T;
}

// Build the BetweenFactor measurement between consecutive world-frame poses.
// The pose estimator returns (R, t) as T_curr_prev (x_curr = R*x_prev + t).
// The world-frame chain X_world(i) = X_world(i-1) * measurement needs the
// inverse, T_prev_curr = [R^T | -R^T * t].
static gtsam::Pose3 relativePoseMeasurement(const cv::Mat& R, const cv::Mat& t) {
    cv::Mat R64, t64;
    R.convertTo(R64, CV_64F);
    t.convertTo(t64, CV_64F);
    cv::Mat R_inv = R64.t();
    cv::Mat t_inv = -R_inv * t64;
    return cvMatToPose3(makeT(R_inv, t_inv));
}

void LocalBundleAdjustment::runOptimization() {
    std::lock_guard<std::mutex> lock(window_mutex_);

    if (local_window_.empty()) return;

    gtsam::NonlinearFactorGraph graph;
    gtsam::Values initial_estimates;

    double fx = K_.at<double>(0, 0), fy = K_.at<double>(1, 1);
    double cx = K_.at<double>(0, 2), cy = K_.at<double>(1, 2);
    gtsam::Cal3_S2::shared_ptr K_gtsam(new gtsam::Cal3_S2(fx, fy, 0.0, cx, cy));

    // Initial estimates: each frame's snapshot T_world goes in as-is. No
    // accumulation, no reference-frame ambiguity.
    for (size_t i = 0; i < local_window_.size(); ++i) {
        if (local_window_[i].T_world.empty()) {
            std::cerr << "[LBA] Frame " << local_window_[i].frame_id
                      << " missing T_world snapshot; aborting optimization." << std::endl;
            return;
        }
        initial_estimates.insert(X(i), cvMatToPose3(local_window_[i].T_world));
    }

    // Gauge anchor: tight prior on the oldest pose at its initial value.
    {
        auto priorNoise = gtsam::noiseModel::Isotropic::Sigma(6, params_.anchor_prior_sigma);
        graph.addPrior(X(0), cvMatToPose3(local_window_[0].T_world), priorNoise);
    }

    // Drift cap: loose prior on the newest pose at its snapshot value.
    // BetweenFactor alone cannot cap cumulative drift across the window.
    // Each factor only sees ~Delta/W deviation when the chain shifts by
    // Delta total, so the chi-squared cost of meter-scale window drift is
    // tiny and easily absorbed by a small SmartFactor improvement. Anchoring
    // the newest pose is what actually keeps a single pass from running off.
    {
        const size_t last = local_window_.size() - 1;
        const double rs = params_.end_prior_rot_sigma;
        const double ts = params_.end_prior_trans_sigma;
        gtsam::Vector6 endSigmas;
        endSigmas << rs, rs, rs, ts, ts, ts;
        auto endNoise = gtsam::noiseModel::Diagonal::Sigmas(endSigmas);
        graph.addPrior(X(last), cvMatToPose3(local_window_[last].T_world), endNoise);
    }

    // BetweenFactor chain: locks the scale and keeps the solver well-posed
    // when SmartFactors degenerate. Stationary frames feed an identity
    // measurement; moving frames feed the (R, t) from the pose estimator.
    gtsam::Vector6 betweenSigmas;
    betweenSigmas << params_.between_rot_sigma, params_.between_rot_sigma, params_.between_rot_sigma,
                     params_.between_trans_sigma, params_.between_trans_sigma, params_.between_trans_sigma;
    auto betweenNoise = gtsam::noiseModel::Diagonal::Sigmas(betweenSigmas);

    gtsam::Vector6 stationarySigmas;
    stationarySigmas << params_.stationary_rot_sigma, params_.stationary_rot_sigma, params_.stationary_rot_sigma,
                        params_.stationary_trans_sigma, params_.stationary_trans_sigma, params_.stationary_trans_sigma;
    auto stationaryNoise = gtsam::noiseModel::Diagonal::Sigmas(stationarySigmas);

    int between_count = 0;
    for (size_t i = 1; i < local_window_.size(); ++i) {
        gtsam::Pose3 measurement;
        bool tight = local_window_[i].is_stationary;
        if (tight || local_window_[i].R.empty() || local_window_[i].t.empty()) {
            measurement = gtsam::Pose3();  // identity
            tight = true;
        } else {
            measurement = relativePoseMeasurement(local_window_[i].R, local_window_[i].t);
        }
        graph.emplace_shared<gtsam::BetweenFactor<gtsam::Pose3>>(
            X(i - 1), X(i), measurement,
            tight ? stationaryNoise : betweenNoise);
        ++between_count;
    }

    // SmartProjectionPoseFactors.
    //   - pixel_sigma: per-observation noise sigma. Kept loose enough that
    //     each factor doesn't dominate the BetweenFactor chain on poor tracks.
    //   - ZERO_ON_DEGENERACY: degenerate landmarks are dropped entirely
    //     rather than inflating the error with no gradient. Without this the
    //     optimizer can sit at err X -> X with a zero correction.
    //   - setDynamicOutlierRejectionThreshold: GTSAM's supported robust path
    //     for SmartFactors. Factors whose post-triangulation reprojection
    //     error exceeds the threshold are treated as degenerate and dropped
    //     under ZERO_ON_DEGENERACY.
    // A Robust m-estimator wrapper is intentionally avoided: HESSIAN
    // linearization Schur-eliminates the landmark, which is incompatible
    // with a per-observation robust kernel.
    auto pixelNoise = gtsam::noiseModel::Isotropic::Sigma(2, params_.pixel_sigma);

    gtsam::SmartProjectionParams smartFactorParams;
    smartFactorParams.setDegeneracyMode(gtsam::DegeneracyMode::ZERO_ON_DEGENERACY);
    smartFactorParams.setRankTolerance(params_.rank_tolerance);
    smartFactorParams.setEnableEPI(false);
    smartFactorParams.setDynamicOutlierRejectionThreshold(params_.outlier_threshold);

    // Group observations by global track id.
    std::map<int64_t, std::vector<std::pair<int, cv::Point2f>>> feature_tracks;
    for (size_t i = 0; i < local_window_.size(); ++i) {
        const auto& frame = local_window_[i];
        for (size_t pt_idx = 0; pt_idx < frame.points2D.size(); ++pt_idx) {
            int64_t t_id = frame.track_ids[pt_idx];
            if (t_id != -1) {
                feature_tracks[t_id].push_back({static_cast<int>(i), frame.points2D[pt_idx]});
            }
        }
    }

    int smart_count = 0;
    const int kMinObs = params_.min_observations;
    const double kMinBboxDiag = params_.min_bbox_diagonal;

    for (const auto& track_pair : feature_tracks) {
        const auto& observations = track_pair.second;
        if (observations.size() < static_cast<size_t>(kMinObs)) continue;

        // Bounding-box diagonal filter: rejects parallax-starved tracks
        // whose observations cluster within a few pixels. These are exactly
        // the degenerate landmarks SmartFactor was designed to skip; the
        // explicit filter just avoids paying the triangulation cost first.
        float min_x = std::numeric_limits<float>::infinity();
        float max_x = -std::numeric_limits<float>::infinity();
        float min_y = std::numeric_limits<float>::infinity();
        float max_y = -std::numeric_limits<float>::infinity();
        for (const auto& obs : observations) {
            min_x = std::min(min_x, obs.second.x);
            max_x = std::max(max_x, obs.second.x);
            min_y = std::min(min_y, obs.second.y);
            max_y = std::max(max_y, obs.second.y);
        }
        double dx = max_x - min_x, dy = max_y - min_y;
        if (std::sqrt(dx * dx + dy * dy) < kMinBboxDiag) continue;

        gtsam::SmartProjectionPoseFactor<gtsam::Cal3_S2>::shared_ptr smart_factor(
            new gtsam::SmartProjectionPoseFactor<gtsam::Cal3_S2>(
                pixelNoise, K_gtsam, smartFactorParams));

        for (const auto& obs : observations) {
            smart_factor->add(gtsam::Point2(obs.second.x, obs.second.y), X(obs.first));
        }
        graph.push_back(smart_factor);
        ++smart_count;
    }

    // Whole optimization is wrapped in try/catch. CheiralityException and
    // other GTSAM throws can fire from optimize() during inner retriangulation
    // or from graph.error() on initial/final residual evaluation; any of them
    // escaping would abort the worker thread.
    try {
        const double err_before = graph.error(initial_estimates);

        gtsam::LevenbergMarquardtParams params;
        gtsam::LevenbergMarquardtOptimizer optimizer(graph, initial_estimates, params);
        gtsam::Values result = optimizer.optimize();
        const double err_after = graph.error(result);

        const size_t last_idx = local_window_.size() - 1;
        gtsam::Pose3 opt_last = result.at<gtsam::Pose3>(X(last_idx));
        cv::Mat T_opt_world = pose3ToCvMat(opt_last);

        cv::Mat T_snap = local_window_[last_idx].T_world;
        cv::Mat delta = T_opt_world * T_snap.inv();
        cv::Mat delta_t = delta(cv::Rect(3, 0, 1, 3));
        double corr_t_norm = cv::norm(delta_t);
        cv::Mat delta_R = delta(cv::Rect(0, 0, 3, 3));
        double trace = delta_R.at<double>(0,0) + delta_R.at<double>(1,1) + delta_R.at<double>(2,2);
        double cos_theta = std::max(-1.0, std::min(1.0, (trace - 1.0) * 0.5));
        double corr_rot_deg = std::acos(cos_theta) * 180.0 / M_PI;

        if (verbose_) {
            std::cout << "[LBA] frame=" << local_window_[last_idx].frame_id
                      << " poses=" << local_window_.size()
                      << " smart=" << smart_count
                      << " between=" << between_count
                      << " err " << err_before << " -> " << err_after
                      << " corr_t=" << corr_t_norm
                      << " corr_rot_deg=" << corr_rot_deg
                      << std::endl;
        }

        // Reject under-determined passes. Too few SmartFactors and the
        // optimizer can drive err to ~zero by shuffling poses freely; the
        // resulting "correction" is noise.
        if (smart_count < params_.min_smart_factors) {
            if (verbose_) {
                std::cerr << "[LBA] discarding correction (under-constrained, smart="
                          << smart_count << ")" << std::endl;
            }
            return;
        }

        // Reject jumps that indicate a bad solution from contaminated tracks.
        // The newest-pose loose prior already caps the optimizer's reach;
        // these thresholds catch anything that still overshoots.
        if (corr_t_norm > params_.max_correction_translation ||
            corr_rot_deg > params_.max_correction_rotation_deg) {
            if (verbose_) {
                std::cerr << "[LBA] discarding correction (out of bounds): "
                          << "corr_t=" << corr_t_norm
                          << " corr_rot_deg=" << corr_rot_deg << std::endl;
            }
            return;
        }

        // Critical: copy the optimized poses back into the window so the
        // LBA's view stays internally consistent on the next pass.
        // Otherwise older frames keep their pre-correction snapshots while
        // newer frames carry the integrator's post-correction state, and
        // the BetweenFactor chain develops a non-zero initial residual
        // equal to the previous correction. The optimizer then partly
        // undoes its own work each cycle. Only safe when this pass is
        // actually publishing; the integrator state and LBA window must
        // stay aligned.
        for (size_t i = 0; i < local_window_.size(); ++i) {
            local_window_[i].T_world = pose3ToCvMat(result.at<gtsam::Pose3>(X(i)));
        }

        BACorrection corr;
        corr.frame_id = local_window_[last_idx].frame_id;
        corr.T_world_optimized = T_opt_world;

        {
            std::lock_guard<std::mutex> lock2(correction_mutex_);
            current_correction_ = corr;
            has_correction_ = true;
        }
    } catch (const std::exception& e) {
        std::cerr << "[LBA] optimization aborted: " << e.what() << std::endl;
        return;
    }
}

bool LocalBundleAdjustment::getCorrection(BACorrection& out_correction) {
    std::lock_guard<std::mutex> lock(correction_mutex_);
    if (has_correction_) {
        out_correction = current_correction_;
        out_correction.T_world_optimized = current_correction_.T_world_optimized.clone();
        has_correction_ = false;
        return true;
    }
    return false;
}
