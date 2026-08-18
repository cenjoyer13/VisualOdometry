#pragma once
#include <opencv2/core.hpp>

// Solves the single Z rotation that takes the estimator's world frame into the
// navigation frame, and holds it for the rest of the run.
//
// WHY ONLY YAW
// ------------
// A visual-inertial estimator observes gravity, so roll and pitch are pinned by
// the accelerometer and its world Z is already up. Yaw is the one degree of
// freedom it cannot observe -- VINS zeroes it at initialisation -- so exactly
// one angle has to come from somewhere else.
//
// TWO WAYS TO GET IT
// ------------------
// GtDisplacement  fit it from ground truth: once GT has travelled
//                 `init_distance` HORIZONTALLY, rotate the estimate's
//                 displacement onto GT's. Evaluation only -- it needs a GT
//                 track, so it cannot run in the field.
//
// Heading         take it from a heading sensor at the first tracked frame:
//                 align_yaw = mount_offset - heading(t0). Deployment-valid: no
//                 GT, no position reference, and it is correct from frame 0
//                 rather than after the first init_distance metres.
//
// `mount_offset` is the constant between the heading source's zero and the
// estimator's yaw datum. It is mechanical plus convention, so it is calibrated
// ONCE on the ground against a known bearing -- never fitted to a flight, which
// would just be the ground-truth dependency wearing a different hat.
//
// A NOTE ON init_distance, LEARNED THE HARD WAY
// ---------------------------------------------
// Distance alone is a poor trigger. On bell412_dataset3 the default 10 m
// accrues 10.9 s after tracking starts at 0.93 m/s horizontal, while the
// helicopter is climbing almost vertically -- so the displacement direction
// comes from near-hover wander and the resulting yaw is ~9 deg off the
// whole-trajectory best fit, which costs more ATE than the trajectory error
// itself. `min_speed` gates on the vehicle actually translating, which is the
// condition that makes a displacement direction mean anything.
class TrajectoryAligner {
public:
    enum class Mode {
        None,            // identity; the estimator's own frame is reported
        GtDisplacement,  // fit from GT displacement (evaluation only)
        Heading          // seed from a heading sensor at t0 (deployment)
    };

    struct Params {
        Mode mode = Mode::GtDisplacement;
        double init_distance = 10.0;   // m of horizontal GT travel (GtDisplacement)
        double min_speed = 0.0;        // m/s horizontal floor before accumulating
        double mount_offset_deg = 0.0; // heading-source zero -> estimator yaw datum
    };

    explicit TrajectoryAligner(const Params& p) : p_(p) {}

    bool ready() const { return ready_; }
    // 4x4 world rotation; identity until ready().
    const cv::Mat& transform() const { return T_; }

    // Feed one frame. `est_xy` and `gt_xy` are horizontal positions (metres);
    // `heading_rad` is the heading source's reading, used only in Heading mode.
    // `dt` is the interval since the previous call, for the speed gate.
    // Returns true on the frame the alignment is solved.
    bool update(const cv::Point2d& est_xy, const cv::Point2d& gt_xy,
                double heading_rad, double dt);

    // Diagnostics for the frame it solved on.
    double solvedYawDeg() const { return yaw_deg_; }
    double travelledAtSolve() const { return travelled_; }

private:
    void setYaw(double yaw_rad);

    Params p_;
    cv::Mat T_ = cv::Mat::eye(4, 4, CV_64F);
    bool ready_ = false;
    bool started_ = false;
    cv::Point2d est_start_{0, 0}, gt_start_{0, 0}, gt_prev_{0, 0};
    double yaw_deg_ = 0.0, travelled_ = 0.0;
};
