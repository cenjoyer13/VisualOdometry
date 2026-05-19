#include "FeatureUtils.h"
#include <numeric>
#include <algorithm>
#include <future>

void FeatureUtils::detectWithGridCPU(const std::function<cv::Ptr<cv::Feature2D>()>& detector_builder,
                                     const cv::Mat& image,
                                     std::vector<cv::KeyPoint>& out_keypoints,
                                     cv::Mat& out_descriptors,
                                     const BucketingConfig& config,
                                     int num_threads) 
{
    int total_rois = config.grid_rows * config.grid_cols;
    
    // Fallback to serial if threads = 1
    int active_threads = std::max(1, std::min(num_threads, total_rois)); 
    int rois_per_thread = total_rois / active_threads;

    using ThreadResult = std::pair<std::vector<cv::KeyPoint>, cv::Mat>;
    std::vector<std::future<ThreadResult>> futures;

    for (int t = 0; t < active_threads; ++t) {
        int start_idx = t * rois_per_thread;
        int end_idx = (t == active_threads - 1) ? total_rois : start_idx + rois_per_thread;

        // Launch async worker
        futures.push_back(std::async(std::launch::async, [start_idx, end_idx, &image, &config, detector_builder]() {
            // Build a thread-local detector to prevent race conditions
            cv::Ptr<cv::Feature2D> local_detector = detector_builder();
            std::vector<cv::KeyPoint> local_kps;
            cv::Mat local_desc;

            int cell_w = image.cols / config.grid_cols;
            int cell_h = image.rows / config.grid_rows;

            for (int i = start_idx; i < end_idx; ++i) {
                int r = i / config.grid_cols;
                int c = i % config.grid_cols;

                cv::Rect roi(c * cell_w, r * cell_h, cell_w, cell_h);
                if (c == config.grid_cols - 1) roi.width = image.cols - roi.x;
                if (r == config.grid_rows - 1) roi.height = image.rows - roi.y;

                cv::Mat cell_img = image(roi);
                std::vector<cv::KeyPoint> cell_kps;
                cv::Mat cell_desc;

                local_detector->detectAndCompute(cell_img, cv::noArray(), cell_kps, cell_desc);

                if (cell_kps.empty()) continue;

                // Sort by response
                std::vector<int> indices(cell_kps.size());
                std::iota(indices.begin(), indices.end(), 0);
                std::sort(indices.begin(), indices.end(), [&cell_kps](int a, int b) {
                    return cell_kps[a].response > cell_kps[b].response;
                });

                int keep = std::min(config.max_features_per_bucket, (int)cell_kps.size());
                for (int j = 0; j < keep; ++j) {
                    int orig_idx = indices[j];
                    cv::KeyPoint kp = cell_kps[orig_idx];
                    kp.pt.x += roi.x; 
                    kp.pt.y += roi.y;
                    local_kps.push_back(kp);

                    if (local_desc.empty()) {
                        local_desc = cell_desc.row(orig_idx).clone();
                    } else {
                        local_desc.push_back(cell_desc.row(orig_idx));
                    }
                }
            }
            return std::make_pair(local_kps, local_desc);
        }));
    }

    // Safely collect and merge all thread results
    for (auto& f : futures) {
        ThreadResult res = f.get();
        out_keypoints.insert(out_keypoints.end(), res.first.begin(), res.first.end());
        if (!res.second.empty()) {
            if (out_descriptors.empty()) out_descriptors = res.second.clone();
            else out_descriptors.push_back(res.second);
        }
    }
}
