#pragma once
#include "ITrajectoryIntegrator.h"
#include "../OdometryTypes.h"
#include <unordered_map>
#include <deque>
#include <cstdint>

// Accumulates per-frame relative (R, t) measurements from the pose estimator
// into a global 4x4 camera-in-world transform. Stores bounded snapshots of
// past poses so LocalBundleAdjustment corrections can be applied as a
// world-frame delta against the snapshot taken at push time.
class VOIntegrator : public ITrajectoryIntegrator {
private:
    OdometryConfig config;

    // 4x4 homogeneous transformation matrix — camera in world frame.
    cv::Mat T_VO;

    // Bounded snapshot store for absolute-pose corrections. Hard cap by count.
    static constexpr size_t kMaxSnapshots = 64;
    std::unordered_map<uint64_t, cv::Mat> snapshots_;
    std::deque<uint64_t> snapshot_order_;

public:
    explicit VOIntegrator(const OdometryConfig& cfg);
    ~VOIntegrator() override = default;

    void snapshotPose(uint64_t frame_id) override;
    bool getSnapshot(uint64_t frame_id, cv::Mat& out_T_world) const override;
    bool applyCorrection(uint64_t frame_id, const cv::Mat& T_world_optimized) override;

    void integrate(const cv::Mat& local_R,
                   const cv::Mat& local_t,
                   double scale) override;

    cv::Mat getGlobalTransformVO() const override;
};
