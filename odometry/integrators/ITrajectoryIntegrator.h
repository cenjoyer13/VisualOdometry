#pragma once
#include <opencv2/core.hpp>
#include <cstdint>
#include "../OdometryTypes.h"

class ITrajectoryIntegrator {
public:
    virtual ~ITrajectoryIntegrator() = default;

    // Record the current global VO pose so a later LBA correction can be
    // computed as a world-frame delta against this exact snapshot.
    virtual void snapshotPose(uint64_t frame_id) = 0;

    // Read back a previously stored snapshot. Returns false if no snapshot
    // exists for that frame_id (evicted or never taken).
    virtual bool getSnapshot(uint64_t frame_id, cv::Mat& out_T_world) const = 0;

    // Absolute-pose correction. The integrator computes delta in world frame
    // against its snapshot taken at frame_id push time, then applies it.
    // Returns false if the snapshot is unavailable (correction is dropped).
    virtual bool applyCorrection(uint64_t frame_id, const cv::Mat& T_world_optimized) = 0;

    // Integrate a new relative pose measurement (R, t) with the given scale
    // into the global trajectory.
    virtual void integrate(const cv::Mat& local_R,
                           const cv::Mat& local_t,
                           double scale) = 0;

    // Returns a 4x4 cv::Mat (CV_64F) representing the full pose:
    // [ R11 R12 R13 tx ]
    // [ R21 R22 R23 ty ]
    // [ R31 R32 R33 tz ]
    // [  0   0   0   1 ]
    virtual cv::Mat getGlobalTransformVO() const = 0;
};
