#pragma once
#include "ITrajectoryIntegrator.h"
#include "../OdometryTypes.h"
#include <unordered_map>
#include <deque>
#include <cstdint>

class DualPathIntegrator : public ITrajectoryIntegrator {
private:
    OdometryConfig config;

    // 4x4 Homogeneous Transformation Matrices
    cv::Mat T_VO;
    cv::Mat T_VIO;

    // Bounded snapshot store for absolute-pose corrections. Hard cap by count.
    static constexpr size_t kMaxSnapshots = 64;
    std::unordered_map<uint64_t, cv::Mat> snapshots_;
    std::deque<uint64_t> snapshot_order_;

    // Helper to convert Euler angles (Pitch, Roll, Yaw) to a 3x3 Rotation Matrix
    cv::Mat eulerToRotationMatrix(const cv::Vec3f& euler) const;

public:
    explicit DualPathIntegrator(const OdometryConfig& cfg);
    ~DualPathIntegrator() override = default;

    void snapshotPose(uint64_t frame_id) override;
    bool getSnapshot(uint64_t frame_id, cv::Mat& out_T_world) const override;
    bool applyCorrection(uint64_t frame_id, const cv::Mat& T_world_optimized) override;

    void integrate(const cv::Mat& local_R,
                   const cv::Mat& local_t,
                   double scale,
                   const cv::Vec3f& imu_orientation) override;

    cv::Mat getGlobalTransformVO() const override;
    cv::Mat getGlobalTransformVIO() const override;
};
