#include "OpticalFlowFrontend.h"
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>
#include <algorithm>

OpticalFlowFrontend::OpticalFlowFrontend(const OdometryConfig& cfg)
    : params_(cfg.optical_flow_params), bucketing_(cfg.bucketing_params) {}

cv::Mat OpticalFlowFrontend::toGray(DeviceBuffer& frame) {
    cv::Mat img = frame.getAsCPU();
    if (img.channels() == 1) return img.clone();
    cv::Mat gray;
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    return gray;
}

void OpticalFlowFrontend::addCorners(const cv::Mat& gray, const cv::Mat& mask) {
    const int cols = std::max(1, bucketing_.grid_cols);
    const int rows = std::max(1, bucketing_.grid_rows);
    const int cw = gray.cols / cols;
    const int ch = gray.rows / rows;

    if (!bucketing_.enabled || cw < 1 || ch < 1) {
        // Global detection up to max_corners.
        int want = params_.max_corners - static_cast<int>(tracks_.size());
        if (want <= 0) return;
        std::vector<cv::Point2f> corners;
        cv::goodFeaturesToTrack(gray, corners, want, params_.quality_level,
                                params_.min_distance, mask);
        for (const auto& c : corners)
            tracks_.push_back(Track{next_id_++, c, c, -1});
        return;
    }

    // Per-cell detection: count live tracks per cell, then fill each up to
    // max_features_per_bucket so corners stay spatially spread. Cells are
    // independent, so the goodFeaturesToTrack calls run in parallel (each into
    // its own output); the merge is serial in cell order so track IDs / order
    // stay deterministic and identical to the single-threaded version.
    const int cap = bucketing_.max_features_per_bucket;
    const int ncells = cols * rows;
    std::vector<int> count(ncells, 0);
    for (const auto& t : tracks_) {
        int gx = std::min(cols - 1, static_cast<int>(t.curr_pos.x) / cw);
        int gy = std::min(rows - 1, static_cast<int>(t.curr_pos.y) / ch);
        if (gx >= 0 && gy >= 0) ++count[gy * cols + gx];
    }

    std::vector<std::vector<cv::Point2f>> cell_out(ncells);
    cv::parallel_for_(cv::Range(0, ncells), [&](const cv::Range& range) {
        for (int idx = range.start; idx < range.end; ++idx) {
            int want = cap - count[idx];
            if (want <= 0) continue;
            int gx = idx % cols, gy = idx / cols;
            int x0 = gx * cw, y0 = gy * ch;
            int w = (gx == cols - 1) ? gray.cols - x0 : cw;
            int h = (gy == rows - 1) ? gray.rows - y0 : ch;
            cv::Rect cell(x0, y0, w, h);
            cv::Mat cell_mask = mask.empty() ? cv::Mat() : mask(cell);
            cv::goodFeaturesToTrack(gray(cell), cell_out[idx], want,
                                    params_.quality_level, params_.min_distance, cell_mask);
        }
    });

    for (int idx = 0; idx < ncells; ++idx) {
        int x0 = (idx % cols) * cw, y0 = (idx / cols) * ch;
        for (const auto& c : cell_out[idx])
            tracks_.push_back(Track{next_id_++,
                                    cv::Point2f(c.x + x0, c.y + y0),
                                    cv::Point2f(c.x + x0, c.y + y0), -1});
    }
}

void OpticalFlowFrontend::initialize(DeviceBuffer& frame) {
    prev_gray_ = toGray(frame);
    tracks_.clear();
    addCorners(prev_gray_, cv::Mat());
}

FrontendResult OpticalFlowFrontend::process(DeviceBuffer& frame) {
    FrontendResult r;
    cv::Mat gray = toGray(frame);

    if (!tracks_.empty() && !prev_gray_.empty()) {
        std::vector<cv::Point2f> prev_pts, next_pts, back_pts;
        prev_pts.reserve(tracks_.size());
        for (const auto& t : tracks_) prev_pts.push_back(t.curr_pos);

        std::vector<uchar> status_f, status_b;
        std::vector<float> err_f, err_b;
        const cv::Size win(params_.win_size, params_.win_size);
        const cv::TermCriteria crit(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01);

        cv::calcOpticalFlowPyrLK(prev_gray_, gray, prev_pts, next_pts,
                                 status_f, err_f, win, params_.max_level, crit);
        cv::calcOpticalFlowPyrLK(gray, prev_gray_, next_pts, back_pts,
                                 status_b, err_b, win, params_.max_level, crit);

        // Keep tracks that survived forward+backward LK with low FB error and
        // landed inside the image; update their position.
        std::vector<Track> kept;
        kept.reserve(tracks_.size());
        for (size_t i = 0; i < tracks_.size(); ++i) {
            if (!status_f[i] || !status_b[i]) continue;
            if (cv::norm(prev_pts[i] - back_pts[i]) > params_.fb_error_threshold) continue;
            const cv::Point2f& np = next_pts[i];
            if (np.x < 0 || np.y < 0 || np.x >= gray.cols || np.y >= gray.rows) continue;
            Track t = tracks_[i];
            t.curr_pos = np;
            kept.push_back(t);
        }
        tracks_ = std::move(kept);
    }
    prev_gray_ = gray;

    // Correspondences: every alive track is anchor->current; all are observations.
    r.pts_prev.reserve(tracks_.size());
    r.pts_curr.reserve(tracks_.size());
    r.points2D.reserve(tracks_.size());
    r.matched_prev_idx.reserve(tracks_.size());
    r.track_ids.reserve(tracks_.size());
    for (const auto& t : tracks_) {
        r.pts_prev.push_back(t.anchor_pos);
        r.pts_curr.push_back(t.curr_pos);
        r.points2D.push_back(t.curr_pos);
        r.matched_prev_idx.push_back(t.prev_kf_idx);
        r.track_ids.push_back(t.id);
    }

    // Debug overlay: bucketing grid + anchor->current flow vectors.
    cv::Mat color;
    cv::cvtColor(gray, color, cv::COLOR_GRAY2BGR);
    if (bucketing_.enabled) {
        int cols = std::max(1, bucketing_.grid_cols);
        int rows = std::max(1, bucketing_.grid_rows);
        float cw = static_cast<float>(color.cols) / cols;
        float ch = static_cast<float>(color.rows) / rows;
        for (int i = 1; i < cols; ++i)
            cv::line(color, cv::Point(static_cast<int>(i * cw), 0),
                     cv::Point(static_cast<int>(i * cw), color.rows), cv::Scalar(255, 50, 50), 1, cv::LINE_AA);
        for (int i = 1; i < rows; ++i)
            cv::line(color, cv::Point(0, static_cast<int>(i * ch)),
                     cv::Point(color.cols, static_cast<int>(i * ch)), cv::Scalar(255, 50, 50), 1, cv::LINE_AA);
    }
    for (const auto& t : tracks_) {
        cv::line(color, t.anchor_pos, t.curr_pos, cv::Scalar(0, 0, 255), 1, cv::LINE_AA);
        cv::circle(color, t.curr_pos, 3, cv::Scalar(0, 255, 0), -1, cv::LINE_AA);
    }
    r.debug_overlay = color;
    return r;
}

void OpticalFlowFrontend::promoteKeyframe(DeviceBuffer& /*frame*/) {
    // The just-processed points2D order == current tracks_ order. Record each
    // track's index in it (for the next keyframe's matched_prev_idx) and reset
    // the baseline to the current position.
    for (size_t i = 0; i < tracks_.size(); ++i) {
        tracks_[i].prev_kf_idx = static_cast<int>(i);
        tracks_[i].anchor_pos = tracks_[i].curr_pos;
    }
    // Replenish corners (new tracks carry prev_kf_idx = -1; not in this keyframe).
    // The mask suppresses a neighborhood around live tracks; addCorners applies
    // the bucketing per-cell cap (or the global max_corners cap).
    if (!prev_gray_.empty()) {
        cv::Mat mask(prev_gray_.size(), CV_8U, cv::Scalar(255));
        for (const auto& t : tracks_)
            cv::circle(mask, t.curr_pos, static_cast<int>(params_.min_distance), 0, -1);
        addCorners(prev_gray_, mask);
    }
}
