#include "DescriptorFrontend.h"
#include <opencv2/imgproc.hpp>
#include <chrono>

DescriptorFrontend::DescriptorFrontend(const OdometryConfig& cfg,
                                       std::unique_ptr<IFeatureDetector> detector,
                                       std::unique_ptr<IFeatureMatcher> matcher)
    : config_(cfg), detector_(std::move(detector)), matcher_(std::move(matcher)) {}

void DescriptorFrontend::initialize(DeviceBuffer& frame) {
    std::vector<cv::KeyPoint> kpts;
    DeviceBuffer desc;
    detector_->detect(frame, kpts, desc);
    anchor_image_ = frame;
    anchor_keypoints_ = kpts;
    anchor_descriptors_ = desc;
}

FrontendResult DescriptorFrontend::process(DeviceBuffer& frame) {
    FrontendResult r;

    // Detection.
    auto t0 = std::chrono::high_resolution_clock::now();
    std::vector<cv::KeyPoint> curr_keypoints;
    DeviceBuffer curr_descriptors;
    detector_->detect(frame, curr_keypoints, curr_descriptors);
    auto t1 = std::chrono::high_resolution_clock::now();
    r.detect_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // Cache the current frame so promoteKeyframe advances to it.
    last_image_ = frame;
    last_keypoints_ = curr_keypoints;
    last_descriptors_ = curr_descriptors;

    // Matching against the anchor.
    t0 = std::chrono::high_resolution_clock::now();
    std::vector<cv::DMatch> good_matches = matcher_->match(
        anchor_descriptors_, curr_descriptors, anchor_keypoints_, curr_keypoints);
    t1 = std::chrono::high_resolution_clock::now();
    r.match_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // matched_prev_idx[curr_idx] = anchor index, or -1 if unmatched.
    r.matched_prev_idx.assign(curr_keypoints.size(), -1);
    for (const auto& m : good_matches) r.matched_prev_idx[m.trainIdx] = m.queryIdx;

    // Lift matches to raw 2D point pairs for the pose estimator.
    r.pts_prev.reserve(good_matches.size());
    r.pts_curr.reserve(good_matches.size());
    for (const auto& m : good_matches) {
        r.pts_prev.push_back(anchor_keypoints_[m.queryIdx].pt);
        r.pts_curr.push_back(curr_keypoints[m.trainIdx].pt);
    }

    // All current observations for the LBA.
    cv::KeyPoint::convert(curr_keypoints, r.points2D);

    // Debug overlay: matched feature tracks on the input frame.
    cv::Mat color_frame;
    cv::Mat cpu_img = frame.getAsCPU();
    if (cpu_img.channels() == 1) {
        cv::cvtColor(cpu_img, color_frame, cv::COLOR_GRAY2BGR);
    } else {
        color_frame = cpu_img.clone();
    }

    if (config_.bucketing_params.enabled) {
        int cols = config_.bucketing_params.grid_cols;
        int rows = config_.bucketing_params.grid_rows;
        int width = color_frame.cols;
        int height = color_frame.rows;
        float cell_w = static_cast<float>(width) / cols;
        float cell_h = static_cast<float>(height) / rows;
        cv::Scalar grid_color(255, 50, 50);
        for (int i = 1; i < cols; ++i) {
            int x = static_cast<int>(i * cell_w);
            cv::line(color_frame, cv::Point(x, 0), cv::Point(x, height), grid_color, 1, cv::LINE_AA);
        }
        for (int i = 1; i < rows; ++i) {
            int y = static_cast<int>(i * cell_h);
            cv::line(color_frame, cv::Point(0, y), cv::Point(width, y), grid_color, 1, cv::LINE_AA);
        }
    }

    for (size_t i = 0; i < r.pts_curr.size(); ++i) {
        cv::line(color_frame, r.pts_prev[i], r.pts_curr[i], cv::Scalar(0, 0, 255), 1, cv::LINE_AA);
        cv::circle(color_frame, r.pts_curr[i], 3, cv::Scalar(0, 255, 0), -1, cv::LINE_AA);
    }
    r.debug_overlay = color_frame;

    return r;
}

void DescriptorFrontend::promoteKeyframe(DeviceBuffer& /*frame*/) {
    anchor_image_ = last_image_;
    anchor_keypoints_ = last_keypoints_;
    anchor_descriptors_ = last_descriptors_;
}
