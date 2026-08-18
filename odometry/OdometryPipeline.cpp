#include "OdometryPipeline.h"
#include <iostream>
#include <chrono>
#include <utility>

#include "frontend/FrontendFactory.h"
#include "utils/RunLog.h"
#include "utils/Perf.h"

std::unique_ptr<OdometryPipeline> OdometryPipeline::build(const OdometryConfig& config,
                                                          const std::string& vins_config,
                                                          const ICameraModel* camera) {
    std::cout << "[OdometryPipeline] Initiating build sequence..." << std::endl;
    return std::make_unique<OdometryPipeline>(
        config,
        FrontendFactory::create(config),
        std::make_unique<VinsBackend>(vins_config),
        camera
    );
}

OdometryPipeline::OdometryPipeline(const OdometryConfig& cfg,
                                   std::unique_ptr<IFrontend> f,
                                   std::unique_ptr<VinsBackend> b,
                                   const ICameraModel* camera)
    : config(cfg),
      frontend(std::move(f)),
      backend_(std::move(b)),
      camera_(camera),
      rejector_(cfg.outlier_rejection)
{
    std::cout << "[OdometryPipeline] Frontend + VINS backend assembled ("
              << (config.undistort_mode == UndistortMode::Points
                      ? "raw frames, points lifted"
                      : "rectified frames, linear normalisation")
              << ")." << std::endl;
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

    FrontendResult fr;
    {
        PERF_SCOPE(perf::Frontend);
        fr = frontend->process(frame);
    }
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

    {
        PERF_SCOPE(perf::Undistort);
        liftPixels(fr.points2D, norm_);
    }

    // Geometric outlier rejection, on bearings rather than pixels -- see
    // OutlierRejector. Requires the correspondence pair, so the previous view's
    // matching features are lifted too.
    //
    // Only runs when the frontend's four output arrays are index-aligned, which
    // is what the backend consumes. Optical flow satisfies that (every live
    // track is a correspondence); the descriptor frontend does not yet
    // (points2D is every keypoint, pts_curr only the matched ones), so it is
    // skipped there rather than silently filtering the wrong indices.
    if (rejector_.enabled() &&
        fr.pts_prev.size() == fr.points2D.size() &&
        fr.pts_curr.size() == fr.points2D.size()) {
        PERF_SCOPE(perf::Reject);
        liftPixels(fr.pts_prev, norm_prev_);
        const auto res = rejector_.run(norm_prev_, norm_, inlier_mask_);

        if (res.ran) {
            dropped_ids_.clear();
            size_t k = 0;
            for (size_t i = 0; i < inlier_mask_.size(); ++i) {
                if (inlier_mask_[i]) {
                    fr.track_ids[k] = fr.track_ids[i];
                    fr.points2D[k]  = fr.points2D[i];
                    norm_[k]        = norm_[i];
                    ++k;
                } else {
                    dropped_ids_.push_back(fr.track_ids[i]);
                }
            }
            fr.track_ids.resize(k);
            fr.points2D.resize(k);
            norm_.resize(k);
            // Retire them in the frontend too, or a bad track survives and is
            // re-rejected every frame while holding a max_corners slot.
            frontend->dropTracks(dropped_ids_);
        }

        LogRec("reject")("ran", res.ran)("n_in", res.n_in)("n_out", res.n_out)
                        ("skip", res.skip_reason);
    }

    const auto t0 = std::chrono::high_resolution_clock::now();
    {
        PERF_SCOPE(perf::Backend);
        backend_->addFrame(timestamp, fr.track_ids, norm_, fr.points2D);
    }
    const auto t1 = std::chrono::high_resolution_clock::now();
    metrics.time_pose_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // Re-anchor and replenish corners every frame, matching VINS's own tracker:
    // it tracks and tops up continuously rather than holding a keyframe anchor.
    // Counted as frontend work -- it is where corner replenishment happens.
    {
        PERF_SCOPE(perf::Frontend);
        frontend->promoteKeyframe(frame);
    }

    const auto t_end = std::chrono::high_resolution_clock::now();
    metrics.time_total_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    metrics.fps = metrics.time_total_ms > 0.0 ? 1000.0 / metrics.time_total_ms : 0.0;
}

void OdometryPipeline::liftPixels(const std::vector<cv::Point2f>& px,
                                  std::vector<cv::Point2d>& out) const {
    // The two undistort modes differ here and nowhere else:
    //   Points -- the frontend saw the RAW frame, so the full lens model has to
    //             be inverted per feature.
    //   Image  -- the evaluator already rectified the frame, so the mapping is
    //             the plain pinhole normalisation against the rectified
    //             intrinsics. Calling liftProjective here would undistort twice.
    if (config.undistort_mode == UndistortMode::Points) {
        camera_->liftProjective(px, out);
        return;
    }
    const CameraIntrinsics& K = config.intrinsics;
    out.resize(px.size());
    for (size_t i = 0; i < px.size(); ++i) {
        out[i].x = (static_cast<double>(px[i].x) - K.cx) / K.fx;
        out[i].y = (static_cast<double>(px[i].y) - K.cy) / K.fy;
    }
}

cv::Mat OdometryPipeline::getGlobalTransformVO() const {
    cv::Mat T;
    if (backend_->latestPose(T)) return T;
    return cv::Mat::eye(4, 4, CV_64F);
}

bool OdometryPipeline::isTrackingActive() const {
    return backend_->isInitialized();
}
