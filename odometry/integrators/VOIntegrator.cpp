#include "VOIntegrator.h"

VOIntegrator::VOIntegrator(const OdometryConfig& cfg) : config(cfg) {
    T_VO = cv::Mat::eye(4, 4, CV_64F);
}

void VOIntegrator::snapshotPose(uint64_t frame_id) {
    snapshots_[frame_id] = T_VO.clone();
    snapshot_order_.push_back(frame_id);
    while (snapshot_order_.size() > kMaxSnapshots) {
        uint64_t evict = snapshot_order_.front();
        snapshot_order_.pop_front();
        snapshots_.erase(evict);
    }
}

bool VOIntegrator::getSnapshot(uint64_t frame_id, cv::Mat& out_T_world) const {
    auto it = snapshots_.find(frame_id);
    if (it == snapshots_.end()) return false;
    out_T_world = it->second.clone();
    return true;
}

bool VOIntegrator::applyCorrection(uint64_t frame_id, const cv::Mat& T_world_optimized) {
    auto it = snapshots_.find(frame_id);
    if (it == snapshots_.end()) {
        // Snapshot evicted; correction is no longer anchorable. Drop it.
        return false;
    }

    cv::Mat T_snap = it->second;
    cv::Mat T_opt;
    T_world_optimized.convertTo(T_opt, CV_64F);

    // Delta in world frame: applying delta to the snapshot recovers T_opt.
    //   T_opt = delta * T_snap   =>   delta = T_opt * T_snap.inv()
    cv::Mat delta = T_opt * T_snap.inv();

    // Left-multiply so any motion accumulated since the snapshot remains
    // correct relative to the corrected pose.
    T_VO = delta * T_VO;

    return true;
}

void VOIntegrator::integrate(const cv::Mat& local_R, const cv::Mat& local_t, double scale) {
    cv::Mat R_64, t_64;
    local_R.convertTo(R_64, CV_64F);
    local_t.convertTo(t_64, CV_64F);

    // Convention: the pose estimator returns (R, t) as T_curr_prev, meaning
    // x_curr = R*x_prev + t. World-frame accumulation needs T_prev_curr, so
    // invert by transposing R and negating R^T * t.
    cv::Mat R_cam = R_64.t();
    cv::Mat t_cam = -R_cam * t_64;
    cv::Mat scaled_t = t_cam * scale;

    cv::Mat T_local = cv::Mat::eye(4, 4, CV_64F);
    R_cam.copyTo(T_local(cv::Rect(0, 0, 3, 3)));
    scaled_t.copyTo(T_local(cv::Rect(3, 0, 1, 3)));

    // T_global = T_global * T_local
    T_VO = T_VO * T_local;
}

cv::Mat VOIntegrator::getGlobalTransformVO() const { return T_VO.clone(); }
