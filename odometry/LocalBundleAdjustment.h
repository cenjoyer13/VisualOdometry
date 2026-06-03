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

#include "OdometryTypes.h"

struct BAFrame {
    uint64_t frame_id = 0;

    // Integrator's world pose at push time. 4x4 CV_64F.
    cv::Mat T_world;

    // Relative pose measurement (T_curr_prev), with t already metric-scaled
    // by the pipeline. Used as the BetweenFactor measurement to the previous
    // frame in the window.
    cv::Mat R;
    cv::Mat t;

    // Stationary tag from the frontend. Stationary frames are still pushed
    // so prev_frame_track_ids_ stays aligned with the frontend match indexing;
    // LBA collapses their BetweenFactor to a tight identity.
    bool is_stationary = false;

    std::vector<cv::Point2f> points2D;

    // Frontend fills this: matching index in the previous frame, or -1 if new.
    std::vector<int> matched_prev_idx;

    // Backend fills this: the global landmark ID assigned per 2D point.
    std::vector<int64_t> track_ids;

    // IMU samples covering (previous keyframe, this keyframe]. Empty unless an
    // IMU mode is active. Used to build the gyro rotation prior (and, later,
    // the CombinedImuFactor).
    std::vector<ImuSample> imu_samples;
};

// Output of one LBA pass: the optimized world-frame pose for the newest
// frame in the window. The integrator turns this into a world-frame delta
// using the snapshot it took at frame_id push time.
struct BACorrection {
    uint64_t frame_id = 0;
    cv::Mat T_world_optimized;  // 4x4 CV_64F
};

class LocalBundleAdjustment {
public:
    LocalBundleAdjustment(const cv::Mat& K, const LBAParams& params,
                          const ImuParams& imu_params, bool verbose = false);
    ~LocalBundleAdjustment();

    void start();
    void stop();
    void pushFrame(const BAFrame& frame);

    bool getCorrection(BACorrection& out_correction);

private:
    void optimizationLoop();
    void runOptimization();

    LBAParams params_;
    ImuParams imu_params_;
    bool verbose_ = false;
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

    int frames_since_last_opt_ = 0;

    std::unordered_map<int64_t, int> active_landmarks_;

    void assignTrackIDs(BAFrame& frame);
    void pruneOutdatedTracks(const BAFrame& old_frame);
};
