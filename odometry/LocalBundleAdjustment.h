#pragma once

#include <opencv2/core.hpp>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <vector>
#include <queue>
#include <optional>
#include <unordered_map>
#include <cstdint>

struct BAFrame {
    uint64_t frame_id = 0;

    // Snapshot of the integrator's world pose at the moment this frame was pushed.
    // 4x4 CV_64F. Phase 1 absolute-pose contract.
    cv::Mat T_world;

    // Relative pose measurement from the pose estimator (T_curr_prev convention,
    // t already scaled to metric by the pipeline). Retained for use as the
    // BetweenFactor measurement between consecutive poses.
    cv::Mat R;
    cv::Mat t;

    // True if the pipeline considered this frame stationary (norm_t below the
    // motion threshold). Pushed anyway so LBA's prev_frame_track_ids_ stays in
    // sync; LBA treats it as a near-identity BetweenFactor.
    bool is_stationary = false;

    std::vector<cv::Point2f> points2D;

    // Frontend fills: index of the matching point in the PREV frame (-1 if new).
    std::vector<int> matched_prev_idx;

    // Backend fills: the global landmark ID per 2D point.
    std::vector<int64_t> track_ids;
};

// Result of a single LBA optimization: the world-frame pose for the newest
// optimized frame, identified by frame_id. The integrator computes its own
// world-frame delta against the snapshot it took at push time.
struct BACorrection {
    uint64_t frame_id = 0;
    cv::Mat T_world_optimized;  // 4x4 CV_64F
};

class LocalBundleAdjustment {
public:
    explicit LocalBundleAdjustment(const cv::Mat& K, int window_size = 10, int opt_stride = 2);
    ~LocalBundleAdjustment();

    void start();
    void stop();
    void pushFrame(const BAFrame& frame);

    bool getCorrection(BACorrection& out_correction);

private:
    void optimizationLoop();
    void runOptimization();

    int window_size_;
    cv::Mat K_;

    std::thread ba_thread_;
    std::mutex queue_mutex_;
    std::mutex window_mutex_;
    std::condition_variable cv_;
    std::atomic<bool> is_running_;

    std::queue<BAFrame> pending_frames_;
    std::vector<BAFrame> local_window_;

    std::mutex correction_mutex_;
    BACorrection current_correction_;
    bool has_correction_ = false;

    int64_t next_track_id_ = 0;
    std::vector<int64_t> prev_frame_track_ids_;

    int opt_stride_;
    int frames_since_last_opt_ = 0;

    std::unordered_map<int64_t, int> active_landmarks_;

    void assignTrackIDs(BAFrame& frame);
    void pruneOutdatedTracks(const BAFrame& old_frame);
};
