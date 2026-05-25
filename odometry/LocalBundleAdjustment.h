#pragma once

#include <opencv2/core.hpp>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <vector>
#include <queue>
#include <optional>

struct BAFrame {
    int frame_id;
    cv::Mat R; 
    cv::Mat t; 
    std::vector<cv::Point2f> points2D;
    
    // Frontend fills this: Index of the matching point in the PREV frame (-1 if new)
    std::vector<int> matched_prev_idx; 
    
    // Backend fills this: The global landmark ID
    std::vector<int64_t> track_ids; 
};

class LocalBundleAdjustment {
public:
    // Pass camera intrinsics (K) for GTSAM Projection Factors
    explicit LocalBundleAdjustment(const cv::Mat& K, int window_size = 10);
    ~LocalBundleAdjustment();

    void start();
    void stop();
    void pushFrame(const BAFrame& frame);
    
    // --- NEW: Method for the main thread to pull the drift correction ---
    bool getCorrection(cv::Mat& out_correction);

private:
    void optimizationLoop();
    void runOptimization();

    int window_size_;
    cv::Mat K_; // Camera intrinsic matrix

    // Threading Synchronization
    std::thread ba_thread_;
    std::mutex queue_mutex_;         
    std::mutex window_mutex_;        
    std::condition_variable cv_;
    std::atomic<bool> is_running_;

    // Data Storage
    std::queue<BAFrame> pending_frames_;
    std::vector<BAFrame> local_window_; 

    // --- NEW: Thread-Safe Correction State ---
    std::mutex correction_mutex_;
    cv::Mat current_correction_;
    bool has_correction_ = false;

    // --- NEW: Backend Landmark Management ---
    int64_t next_track_id_ = 0;
    std::vector<int64_t> prev_frame_track_ids_; 

    // --- NEW: Stride variables to prevent delta avalanche ---
    int opt_stride_; 
    int frames_since_last_opt_ = 0;
    
    // Reference Counter: Track ID -> Number of times seen in current window
    std::unordered_map<int64_t, int> active_landmarks_; 

    void assignTrackIDs(BAFrame& frame);
    void pruneOutdatedTracks(const BAFrame& old_frame);
};