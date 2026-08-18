#include "OdometryPipeline.h"
#include <iostream>
#include <chrono>
#include <utility>

#include "frontend/FrontendFactory.h"

std::unique_ptr<OdometryPipeline> OdometryPipeline::build(const OdometryConfig& config,
                                                          const std::string& vins_config) {
    std::cout << "[OdometryPipeline] Initiating build sequence..." << std::endl;
    return std::make_unique<OdometryPipeline>(
        config,
        FrontendFactory::create(config),
        std::make_unique<VinsBackend>(vins_config)
    );
}

OdometryPipeline::OdometryPipeline(const OdometryConfig& cfg,
                                   std::unique_ptr<IFrontend> f,
                                   std::unique_ptr<VinsBackend> b)
    : config(cfg),
      frontend(std::move(f)),
      backend_(std::move(b))
{
    std::cout << "[OdometryPipeline] Frontend + VINS backend assembled." << std::endl;
}

OdometryPipeline::~OdometryPipeline() = default;

void OdometryPipeline::addImu(double t, const cv::Vec3d& acc, const cv::Vec3d& gyr) {
    backend_->addImu(t, acc, gyr);
}

void OdometryPipeline::processFrame(DeviceBuffer& frame, const GroundTruthData& current_gt) {
    processFrame(frame, current_gt, static_cast<double>(synthetic_frame_counter_++));
}

void OdometryPipeline::processFrame(DeviceBuffer& frame,
                                    const GroundTruthData& /*current_gt*/,
                                    double timestamp) {
    const auto t_start = std::chrono::high_resolution_clock::now();

    if (is_first_frame) {
        // Seed the tracker. There are no correspondences on frame 0, so there
        // is nothing to feed the backend; VINS is happy to start at frame 1.
        frontend->initialize(frame);
        is_first_frame = false;
        debug_frame = frame.getAsCPU().clone();
        return;
    }

    FrontendResult fr = frontend->process(frame);
    metrics.time_detect_ms = fr.detect_ms;
    metrics.time_match_ms = fr.match_ms;
    debug_frame = fr.debug_overlay;

    // Persistent track ids are mandatory: VINS keys its inverse-depth landmarks
    // and its parallax-based keyframe decision on them. OpticalFlowFrontend
    // fills track_ids; the descriptor frontend does not, and would need a track
    // manager first. Fail loudly rather than silently feeding VINS a stream of
    // one-observation features it can never triangulate.
    if (fr.track_ids.size() != fr.points2D.size()) {
        if (!warned_no_track_ids_) {
            std::cerr << "[OdometryPipeline] frontend produced no persistent track ids ("
                      << fr.track_ids.size() << " ids vs " << fr.points2D.size()
                      << " points). VINS needs tracks; use frontend: optical_flow "
                         "until a track manager exists.\n";
            warned_no_track_ids_ = true;
        }
        return;
    }

    const auto t0 = std::chrono::high_resolution_clock::now();
    backend_->addFrame(timestamp, fr.track_ids, fr.points2D, config.intrinsics);
    const auto t1 = std::chrono::high_resolution_clock::now();
    metrics.time_pose_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // Re-anchor and replenish corners every frame, matching VINS's own tracker:
    // it tracks and tops up continuously rather than holding a keyframe anchor.
    frontend->promoteKeyframe(frame);

    const auto t_end = std::chrono::high_resolution_clock::now();
    metrics.time_total_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    metrics.fps = metrics.time_total_ms > 0.0 ? 1000.0 / metrics.time_total_ms : 0.0;
}

cv::Mat OdometryPipeline::getGlobalTransformVO() const {
    cv::Mat T;
    if (backend_->latestPose(T)) return T;
    return cv::Mat::eye(4, 4, CV_64F);
}

bool OdometryPipeline::isTrackingActive() const {
    return backend_->isInitialized();
}
