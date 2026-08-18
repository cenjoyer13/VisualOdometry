#pragma once
#include <memory>
#include <string>
#include <cstdint>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include "OdometryTypes.h"
#include "frontend/IFrontend.h"
#include "vins/VinsBackend.h"
#include "camera/ICameraModel.h"

// Frontend + VINS backend.
//
// This replaces the previous estimate-relative-pose / rescale-with-ground-truth
// / integrate / smooth-with-GTSAM stack wholesale. The old chain could not
// estimate metric scale at all -- it read it from GroundTruthData (or an
// altimeter via the Homography path), which is why adding an IMU to it never
// improved anything: the IMU's one irreplaceable contribution was already being
// supplied by a cheat, and the pose estimator's normalized translation had
// thrown the metric content away before any fusion could see it.
//
// Now the pipeline owns only "how correspondences are produced" and hands them
// to VINS, which owns pose, scale, gravity, biases and marginalization. That
// makes the frontend the experimental variable, and -- because scale is no
// longer fed in -- makes frontend comparisons actually meaningful.
//
// Keyframing is VINS's decision now (parallax-based, inside FeatureManager), so
// every processed frame is forwarded. promoteKeyframe() is still called every
// frame, which is exactly how VINS's own tracker behaves: it re-anchors and
// replenishes corners continuously rather than holding an anchor.
class OdometryPipeline {
public:
    ~OdometryPipeline();

    // `vins_config` is a stock VINS-Fusion yaml; see VinsBackend.
    // `camera` must outlive the pipeline. It is what turns feature pixels into
    // bearings, and which of its two mappings is used depends on
    // config.undistort_mode -- see processFrame.
    static std::unique_ptr<OdometryPipeline> build(const OdometryConfig& config,
                                                   const std::string& vins_config,
                                                   const ICameraModel* camera);

    OdometryPipeline(const OdometryConfig& cfg,
                     std::unique_ptr<IFrontend> f,
                     std::unique_ptr<VinsBackend> b,
                     const ICameraModel* camera);

    // `timestamp` is the image time on the IMU clock (seconds). GroundTruthData
    // is accepted for logging/evaluation only -- nothing in the estimate reads
    // it. The 2-arg overload exists for callers with no clock; it synthesizes a
    // monotonic timestamp and is useless for VIO (the IMU could not be
    // associated), so it is kept only to keep non-VIO tools compiling.
    void processFrame(DeviceBuffer& frame, const GroundTruthData& current_gt, double timestamp);
    void processFrame(DeviceBuffer& frame, const GroundTruthData& current_gt);

    // IMU sample on the IMU clock. Units m/s^2 and rad/s. Forwarded straight to
    // VINS, which owns buffering and preintegration.
    void addImu(double t, const cv::Vec3d& acc, const cv::Vec3d& gyr);

    // Latest optimized body pose in the world frame (4x4 CV_64F). Identity
    // until VINS finishes initialization.
    cv::Mat getGlobalTransformVO() const;

    // True once VINS is running its nonlinear solver, i.e. the pose is real.
    bool isTrackingActive() const;

    const PipelineMetrics& getMetrics() const { return metrics; }
    cv::Mat getDebugFrame() const { return debug_frame; }
    const VinsBackend& backend() const { return *backend_; }

private:
    OdometryConfig config;

    PipelineMetrics metrics;
    cv::Mat debug_frame;

    std::unique_ptr<IFrontend> frontend;
    std::unique_ptr<VinsBackend> backend_;
    const ICameraModel* camera_ = nullptr;

    // Scratch for the pixel -> bearing conversion, reused across frames.
    std::vector<cv::Point2f> norm_;

    bool is_first_frame = true;
    bool warned_no_track_ids_ = false;
    uint64_t synthetic_frame_counter_ = 0;
};
