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

    // Clamp to one thread per ROI; degenerates to serial when num_threads is 1.
    int active_threads = std::max(1, std::min(num_threads, total_rois));
    int rois_per_thread = total_rois / active_threads;

    using ThreadResult = std::pair<std::vector<cv::KeyPoint>, cv::Mat>;
    std::vector<std::future<ThreadResult>> futures;

    for (int t = 0; t < active_threads; ++t) {
        int start_idx = t * rois_per_thread;
        int end_idx = (t == active_threads - 1) ? total_rois : start_idx + rois_per_thread;

        futures.push_back(std::async(std::launch::async, [start_idx, end_idx, &image, &config, detector_builder]() {
            // cv::Feature2D instances are not thread-safe: each worker gets
            // its own via the builder callback.
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

                // Rank by response and keep the top-N per cell.
                std::vector<int> indices(cell_kps.size());
                std::iota(indices.begin(), indices.end(), 0);
                std::sort(indices.begin(), indices.end(), [&cell_kps](int a, int b) {
                    return cell_kps[a].response > cell_kps[b].response;
                });

                int keep = std::min(config.max_features_per_bucket, (int)cell_kps.size());
                for (int j = 0; j < keep; ++j) {
                    int orig_idx = indices[j];
                    cv::KeyPoint kp = cell_kps[orig_idx];
                    // Translate from ROI-local to full-image coordinates.
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

    // Merge per-worker results into the output.
    for (auto& f : futures) {
        ThreadResult res = f.get();
        out_keypoints.insert(out_keypoints.end(), res.first.begin(), res.first.end());
        if (!res.second.empty()) {
            if (out_descriptors.empty()) out_descriptors = res.second.clone();
            else out_descriptors.push_back(res.second);
        }
    }
}

void FeatureUtils::filterByGrid(
    const std::vector<cv::KeyPoint>& in_kpts,
    const cv::Mat& in_desc,
    std::vector<cv::KeyPoint>& out_kpts,
    cv::Mat& out_desc,
    int image_width, int image_height,
    const BucketingConfig& params)
{
    int cols = params.grid_cols;
    int rows = params.grid_rows;
    int max_per_bucket = params.max_features_per_bucket;

    float cell_w = static_cast<float>(image_width) / cols;
    float cell_h = static_cast<float>(image_height) / rows;

    // Per-cell index buckets.
    std::vector<std::vector<std::vector<int>>> grid(rows, std::vector<std::vector<int>>(cols));

    for (int i = 0; i < in_kpts.size(); ++i) {
        int col_idx = std::min(static_cast<int>(in_kpts[i].pt.x / cell_w), cols - 1);
        int row_idx = std::min(static_cast<int>(in_kpts[i].pt.y / cell_h), rows - 1);
        grid[row_idx][col_idx].push_back(i);
    }

    out_kpts.clear();
    std::vector<int> final_indices;

    // Per cell: rank by response, keep top-N. Index lists carry the survivors
    // so the descriptor matrix can be reassembled in one pass below.
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            auto& cell_indices = grid[r][c];

            std::sort(cell_indices.begin(), cell_indices.end(), [&](int a, int b) {
                return in_kpts[a].response > in_kpts[b].response;
            });

            int keep_count = std::min(static_cast<int>(cell_indices.size()), max_per_bucket);
            for (int i = 0; i < keep_count; ++i) {
                int original_idx = cell_indices[i];
                out_kpts.push_back(in_kpts[original_idx]);
                final_indices.push_back(original_idx);
            }
        }
    }

    // Rebuild a dense descriptor matrix holding only the surviving rows.
    out_desc = cv::Mat(final_indices.size(), in_desc.cols, in_desc.type());
    for (size_t i = 0; i < final_indices.size(); ++i) {
        in_desc.row(final_indices[i]).copyTo(out_desc.row(i));
    }
}
