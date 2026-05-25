#include "LocalBundleAdjustment.h"
#include <iostream>

// --- GTSAM HEADERS ---
#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Cal3_S2.h>
#include <gtsam/slam/SmartProjectionPoseFactor.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/base/Vector.h>

using gtsam::symbol_shorthand::X; // For camera poses: X(0), X(1), etc.

LocalBundleAdjustment::LocalBundleAdjustment(const cv::Mat& K, int window_size)
    : K_(K.clone()), window_size_(window_size), opt_stride_(window_size), is_running_(false) {}

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

        // 1. Assign Global IDs
        assignTrackIDs(current_frame);

        {
            std::lock_guard<std::mutex> lock(window_mutex_);
            local_window_.push_back(current_frame);
            
            // --- NEW: Increment the stride counter ---
            frames_since_last_opt_++;

            // 2. Sliding Window Memory Cleanup
            if (local_window_.size() > window_size_) {
                pruneOutdatedTracks(local_window_.front());
                local_window_.erase(local_window_.begin());
            }
        }
        
        // --- NEW: Only optimize if the window is full AND the stride is reached ---
        if (local_window_.size() == window_size_ && frames_since_last_opt_ >= opt_stride_) {
            
            runOptimization();
            
            // Reset the counter so it waits for the next discrete chunk of 10 frames
            frames_since_last_opt_ = 0; 
        }
    }
}

// --- LANDMARK MANAGEMENT IMPLEMENTATION ---

void LocalBundleAdjustment::assignTrackIDs(BAFrame& frame) {
    frame.track_ids.assign(frame.points2D.size(), -1);

    for (size_t i = 0; i < frame.points2D.size(); ++i) {
        int prev_idx = frame.matched_prev_idx[i];
        
        // If the frontend found a match, inherit the backend's ID for that track
        if (prev_idx != -1 && prev_idx < prev_frame_track_ids_.size() && prev_frame_track_ids_[prev_idx] != -1) {
            frame.track_ids[i] = prev_frame_track_ids_[prev_idx];
        } else {
            // No match found by frontend: Mint a brand new 3D Landmark ID
            frame.track_ids[i] = next_track_id_++;
        }
        
        // Increment observation count for memory management
        active_landmarks_[frame.track_ids[i]]++;
    }
    
    // Save state to link the next incoming frame
    prev_frame_track_ids_ = frame.track_ids;
}

void LocalBundleAdjustment::pruneOutdatedTracks(const BAFrame& old_frame) {
    for (uint64_t t_id : old_frame.track_ids) {
        if (t_id != -1) {
            active_landmarks_[t_id]--;
            
            // DYNAMIC DELETION: The moment a landmark is no longer in the window, purge it.
            if (active_landmarks_[t_id] <= 0) {
                active_landmarks_.erase(t_id); 
            }
        }
    }
}

// Helper: Convert OpenCV R,t to 4x4 matrix
cv::Mat makeT(const cv::Mat& R, const cv::Mat& t) {
    cv::Mat T = cv::Mat::eye(4, 4, CV_64F);
    R.convertTo(T(cv::Rect(0, 0, 3, 3)), CV_64F);
    t.convertTo(T(cv::Rect(3, 0, 1, 3)), CV_64F);
    return T;
}

void LocalBundleAdjustment::runOptimization() {
    std::lock_guard<std::mutex> lock(window_mutex_);

    gtsam::NonlinearFactorGraph graph;
    gtsam::Values initial_estimates;

    // 1. Setup GTSAM Camera Intrinsics
    double fx = K_.at<double>(0, 0), fy = K_.at<double>(1, 1);
    double cx = K_.at<double>(0, 2), cy = K_.at<double>(1, 2);
    gtsam::Cal3_S2::shared_ptr K_gtsam(new gtsam::Cal3_S2(fx, fy, 0.0, cx, cy));

    // 2. Build Initial Guesses & Calculate Raw Accumulated Pose
    cv::Mat T_raw_accum = cv::Mat::eye(4, 4, CV_64F);
    
    for (size_t i = 0; i < local_window_.size(); ++i) {
        if (i == 0) {
            // Anchor the first frame at Identity
            initial_estimates.insert(X(i), gtsam::Pose3());
            auto priorNoise = gtsam::noiseModel::Isotropic::Sigma(6, 1e-4);
            graph.addPrior(X(i), gtsam::Pose3(), priorNoise);
        } else {
            // Accumulate Raw Transforms
            cv::Mat T_rel = makeT(local_window_[i].R, local_window_[i].t);
            T_raw_accum = T_raw_accum * T_rel.inv(); // Standard visual odometry integration
            
            // Convert to GTSAM Pose3
            gtsam::Rot3 gR(
                T_raw_accum.at<double>(0,0), T_raw_accum.at<double>(0,1), T_raw_accum.at<double>(0,2),
                T_raw_accum.at<double>(1,0), T_raw_accum.at<double>(1,1), T_raw_accum.at<double>(1,2),
                T_raw_accum.at<double>(2,0), T_raw_accum.at<double>(2,1), T_raw_accum.at<double>(2,2)
            );
            gtsam::Point3 gt(T_raw_accum.at<double>(0,3), T_raw_accum.at<double>(1,3), T_raw_accum.at<double>(2,3));
            
            initial_estimates.insert(X(i), gtsam::Pose3(gR, gt));
        }
    }

    // 3. IMPLEMENTED: Add Measurements (Smart Factors)
    
    // a. Define pixel noise (e.g., 1.0 pixel standard deviation)
    auto pixelNoise = gtsam::noiseModel::Isotropic::Sigma(2, 1.0); 

    // b. Setup Smart Factor Parameters
    auto smartFactorParams = gtsam::SmartProjectionParams();
    smartFactorParams.setDegeneracyMode(gtsam::DegeneracyMode::ZERO_ON_DEGENERACY);
    smartFactorParams.setRankTolerance(1e-9);
    smartFactorParams.setEnableEPI(false); // Can be set to true if you want Epipolar checks

    // c. Group all 2D points by their global track ID
    // Map format: track_id -> vector of <frame_index, Point2f>
    std::map<int, std::vector<std::pair<int, cv::Point2f>>> feature_tracks;

    for (size_t i = 0; i < local_window_.size(); ++i) {
        const auto& frame = local_window_[i];
        for (size_t pt_idx = 0; pt_idx < frame.points2D.size(); ++pt_idx) {
            int t_id = frame.track_ids[pt_idx];
            if (t_id != -1) { // Ignore untracked points
                feature_tracks[t_id].push_back({i, frame.points2D[pt_idx]});
            }
        }
    }

    // d. Create a Smart Factor for each tracked feature
    for (const auto& track_pair : feature_tracks) {
        const auto& observations = track_pair.second;

        // A landmark must be observed in at least 2 frames to constrain the geometry
        if (observations.size() >= 2) {
            
            // Create the smart factor using standard pointer allocation (avoids the boost error)
            gtsam::SmartProjectionPoseFactor<gtsam::Cal3_S2>::shared_ptr smart_factor(
                new gtsam::SmartProjectionPoseFactor<gtsam::Cal3_S2>(pixelNoise, K_gtsam, smartFactorParams)
            );

            // Add all (u,v) pixel coordinates across the sliding window to this single factor
            for (const auto& obs : observations) {
                int frame_idx = obs.first;             // The relative frame index (0 to window_size)
                cv::Point2f pt = obs.second;           // The 2D pixel coordinate
                
                smart_factor->add(gtsam::Point2(pt.x, pt.y), X(frame_idx));
            }

            // Push the factor to the graph
            graph.push_back(smart_factor);
        }
    }

    // 4. Optimize Graph
    gtsam::LevenbergMarquardtParams params;
    gtsam::LevenbergMarquardtOptimizer optimizer(graph, initial_estimates, params);
    gtsam::Values result;
    try {
        result = optimizer.optimize();
    } catch (std::exception& e) {
        std::cerr << "[LBA] GTSAM Optimization failed: " << e.what() << std::endl;
        return;
    }

    // 5. Calculate Drift Correction Delta
    // T_correction = T_opt_last * T_raw_last^-1
    gtsam::Pose3 opt_last_pose = result.at<gtsam::Pose3>(X(local_window_.size() - 1));
    gtsam::Matrix4 gtsam_opt_mat = opt_last_pose.matrix();
    
    cv::Mat T_opt_last(4, 4, CV_64F);
    for(int r = 0; r < 4; r++)
        for(int c = 0; c < 4; c++)
            T_opt_last.at<double>(r, c) = gtsam_opt_mat(r, c);

    cv::Mat T_correction = T_opt_last * T_raw_accum.inv();

    // 6. Push correction safely to main thread
    {
        std::lock_guard<std::mutex> lock2(correction_mutex_);
        current_correction_ = T_correction.clone();
        has_correction_ = true;
    }
}

bool LocalBundleAdjustment::getCorrection(cv::Mat& out_correction) {
    std::lock_guard<std::mutex> lock(correction_mutex_);
    if (has_correction_) {
        out_correction = current_correction_.clone();
        has_correction_ = false; // Consume correction
        return true;
    }
    return false;
}