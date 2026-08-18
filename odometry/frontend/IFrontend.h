#pragma once
#include <vector>
#include <cstdint>
#include <opencv2/core.hpp>
#include "../OdometryTypes.h"

// One frame's worth of correspondences against the current keyframe anchor,
// plus the observations the LBA backend needs. Produced by IFrontend::process.
struct FrontendResult {
    std::vector<cv::Point2f> pts_prev;        // matched in the keyframe anchor
    std::vector<cv::Point2f> pts_curr;        // matched in the current frame
    std::vector<cv::Point2f> points2D;        // all current observations (LBA)
    std::vector<int> matched_prev_idx;        // curr idx -> prev idx, -1 if new
    std::vector<int64_t> track_ids;           // persistent IDs (optical-flow only; OF-2)
    cv::Mat debug_overlay;                     // tracks viz for the debug window
    double detect_ms = 0.0;
    double match_ms = 0.0;
};

// The frontend owns "how correspondences are produced" and the keyframe anchor.
// It does NOT decide keyframes: the pipeline calls promoteKeyframe() when its
// keyframe logic fires. Two implementations: DescriptorFrontend (detect+match,
// today's behavior) and OpticalFlowFrontend (Shi-Tomasi + KLT, added in OF-1).
class IFrontend {
public:
    virtual ~IFrontend() = default;

    // First frame: detect and store the initial anchor. No correspondences.
    virtual void initialize(DeviceBuffer& frame) = 0;

    // Produce correspondences of the current frame against the anchor. Does not
    // advance the anchor.
    virtual FrontendResult process(DeviceBuffer& frame) = 0;

    // Promote the frame just processed to the new keyframe anchor.
    virtual void promoteKeyframe(DeviceBuffer& frame) = 0;

    // Retire tracks the pipeline's geometric check rejected. Default no-op,
    // which is correct for any frontend that rebuilds its correspondences from
    // scratch each frame (descriptor matching). A tracking frontend must
    // implement it: otherwise a bad track survives, gets re-rejected every
    // frame, and permanently occupies one of the max_corners slots.
    virtual void dropTracks(const std::vector<int64_t>& /*ids*/) {}
};
