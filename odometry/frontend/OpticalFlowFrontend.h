#pragma once
#include <vector>
#include <cstdint>
#include "IFrontend.h"

// Shi-Tomasi + KLT optical-flow frontend. Tracks corners frame-to-frame (robust
// short baseline) with persistent IDs, and reports anchor->current
// correspondences via track continuity so the pipeline's keyframe/parallax logic
// is unchanged. Corners are replenished on promoteKeyframe (so every track in an
// interval shares the keyframe baseline, keeping the pose estimate consistent);
// the pipeline's keyframe caps force a keyframe when tracks thin.
class OpticalFlowFrontend : public IFrontend {
public:
    explicit OpticalFlowFrontend(const OdometryConfig& cfg);

    void initialize(DeviceBuffer& frame) override;
    FrontendResult process(DeviceBuffer& frame) override;
    void promoteKeyframe(DeviceBuffer& frame) override;

private:
    struct Track {
        int64_t id;
        cv::Point2f anchor_pos;   // position at the current keyframe
        cv::Point2f curr_pos;     // latest tracked position
        int prev_kf_idx;          // index in the previous keyframe's points2D, -1 if new
    };

    static cv::Mat toGray(DeviceBuffer& frame);
    // Append fresh corners. With bucketing enabled, detects per grid cell up to
    // max_features_per_bucket (counting existing tracks in the cell) so corners
    // stay spatially distributed; otherwise a single global detection up to
    // max_corners. The mask suppresses regions (e.g. around live tracks).
    void addCorners(const cv::Mat& gray, const cv::Mat& mask);

    OpticalFlowParams params_;
    BucketingConfig bucketing_;
    cv::Mat prev_gray_;
    std::vector<Track> tracks_;
    int64_t next_id_ = 0;
};
