// MatchDebugger — standalone frontend sanity check.
//
// Runs the configured detector + matcher on pairs of frames from an image
// folder and writes one annotated PNG per pair: the two frames laid out
// side-by-side (or stacked) with every match drawn as a line between its two
// keypoints. Optional RANSAC verification colours the lines by inlier status,
// which is what actually distinguishes "matched a lot" from "matched well".
//
// Nothing here touches OdometryPipeline: the point is to look at the raw
// correspondence field before pose estimation gets a chance to hide bad data.

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/utils/logger.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "odometry/OdometryTypes.h"
#include "odometry/camera/CameraModelConfig.h"
#include "odometry/camera/CameraModelFactory.h"
#include "odometry/camera/ICameraModel.h"
#include "odometry/detectors/DetectorFactory.h"
#include "odometry/detectors/IFeatureDetector.h"
#include "odometry/matchers/IFeatureMatcher.h"
#include "odometry/matchers/MatcherFactory.h"
#include "odometry/utils/ConfigLoader.h"
#ifdef USE_ONNX
#include "odometry/utils/CudaPreload.h"
#endif

namespace {

enum class Layout { SideBySide, Stacked };
enum class Verify { None, Homography, Fundamental };

struct Options {
    std::string yaml_file;
    std::string image_dir;
    std::string out_dir = "match_debug";
    std::vector<std::pair<int, int>> pairs;   // explicit --pairs entries
    int start = 0;
    int count = 4;
    int gap = 1;                              // frames between the two members of a pair
    int stride = 100;                         // frames between consecutive pairs
    Layout layout = Layout::SideBySide;
    Verify verify = Verify::Homography;
    double ransac_thresh = 3.0;
    int max_lines = 0;                        // 0 = draw every match
    bool draw_unmatched = true;
    bool undistort = true;
    bool debug = false;
    std::string detector_override;
    std::string matcher_override;
};

void printUsage() {
    std::cerr <<
        "Usage: ./MatchDebugger [<yaml_config>] [options]\n"
        "\n"
        "Renders detector+matcher correspondences between frame pairs as PNGs.\n"
        "The YAML is the ordinary OdometryConfig schema; only the detector /\n"
        "matcher / bucketing / camera / system blocks are read, plus\n"
        "image_sequence.path as the default image folder.\n"
        "\n"
        "  --images <dir>        folder of frame_NNNNNN.jpg (overrides image_sequence.path)\n"
        "  --out <dir>           output folder for the PNGs (default: match_debug)\n"
        "  --pairs a:b,c:d       explicit frame pairs; overrides --start/--count\n"
        "  --start <n>           first frame index of the first pair (default 0)\n"
        "  --count <n>           number of pairs to render (default 4)\n"
        "  --gap <n>             frame offset inside a pair (default 1 = adjacent)\n"
        "  --stride <n>          frame offset between consecutive pairs (default 100)\n"
        "  --layout side|stack   pair arrangement (default side)\n"
        "  --verify h|f|none     RANSAC model for inlier colouring (default h = homography)\n"
        "  --ransac-thresh <px>  RANSAC reprojection threshold (default 3.0)\n"
        "  --max-lines <n>       cap drawn matches (0 = all, default 0)\n"
        "  --no-unmatched        hide the unmatched keypoint dots\n"
        "  --no-undistort        skip the camera model's undistortion\n"
        "  --detector <type>     override detector.type (ORB|SIFT|SuperPoint)\n"
        "  --matcher <type>      override matcher.type (Kinematic|FLANN|LightGlue)\n"
        "  --debug               verbose detector/matcher logging\n";
}

bool parseArgs(int argc, char** argv, Options& opt) {
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&](std::string& dst) { if (i + 1 < argc) dst = argv[++i]; };

        if (a == "--images" && i + 1 < argc)              next(opt.image_dir);
        else if (a == "--out" && i + 1 < argc)            next(opt.out_dir);
        else if (a == "--detector" && i + 1 < argc)       next(opt.detector_override);
        else if (a == "--matcher" && i + 1 < argc)        next(opt.matcher_override);
        else if (a == "--start" && i + 1 < argc)          opt.start = std::stoi(argv[++i]);
        else if (a == "--count" && i + 1 < argc)          opt.count = std::stoi(argv[++i]);
        else if (a == "--gap" && i + 1 < argc)            opt.gap = std::stoi(argv[++i]);
        else if (a == "--stride" && i + 1 < argc)         opt.stride = std::stoi(argv[++i]);
        else if (a == "--max-lines" && i + 1 < argc)      opt.max_lines = std::stoi(argv[++i]);
        else if (a == "--ransac-thresh" && i + 1 < argc)  opt.ransac_thresh = std::stod(argv[++i]);
        else if (a == "--no-unmatched")                   opt.draw_unmatched = false;
        else if (a == "--no-undistort")                   opt.undistort = false;
        else if (a == "--debug")                          opt.debug = true;
        else if (a == "--layout" && i + 1 < argc) {
            std::string v = argv[++i];
            opt.layout = (v == "stack" || v == "stacked") ? Layout::Stacked : Layout::SideBySide;
        } else if (a == "--verify" && i + 1 < argc) {
            std::string v = argv[++i];
            if (v == "none" || v == "off")            opt.verify = Verify::None;
            else if (v == "f" || v == "fundamental")  opt.verify = Verify::Fundamental;
            else                                      opt.verify = Verify::Homography;
        } else if (a == "--pairs" && i + 1 < argc) {
            std::stringstream ss(argv[++i]);
            std::string tok;
            while (std::getline(ss, tok, ',')) {
                size_t colon = tok.find(':');
                if (colon == std::string::npos) {
                    std::cerr << "Bad --pairs entry '" << tok << "' (expected a:b).\n";
                    return false;
                }
                opt.pairs.emplace_back(std::stoi(tok.substr(0, colon)),
                                       std::stoi(tok.substr(colon + 1)));
            }
        } else if (a == "--help" || a == "-h") {
            return false;
        } else if (a.rfind("--", 0) == 0) {
            std::cerr << "Unknown option '" << a << "'.\n";
            return false;
        } else if (opt.yaml_file.empty()) {
            opt.yaml_file = a;
        }
    }
    return true;
}

std::string framePath(const std::string& dir, int frame_id) {
    char path[1024];
    snprintf(path, sizeof(path), "%s/frame_%06d.jpg", dir.c_str(), frame_id);
    return path;
}

// Per-pair numbers, printed to the console and burned into the image header.
struct PairStats {
    int kp_a = 0, kp_b = 0;
    int matches = 0;
    int inliers = 0;
    double detect_ms = 0.0;
    double match_ms = 0.0;
    double median_disp = 0.0;   // median matched-point displacement (px)
};

// RANSAC verification over the matched point pairs. Returns a per-match inlier
// mask; an all-true mask when verification is off or there are too few points
// for the chosen model (so callers can treat "unverified" as "drawn plain").
std::vector<uchar> verifyMatches(const std::vector<cv::Point2f>& pa,
                                 const std::vector<cv::Point2f>& pb,
                                 Verify mode, double thresh, bool& out_ran) {
    out_ran = false;
    std::vector<uchar> mask(pa.size(), 1);
    if (mode == Verify::None) return mask;

    const size_t min_pts = (mode == Verify::Homography) ? 4 : 8;
    if (pa.size() < min_pts) return mask;

    cv::Mat cv_mask;
    if (mode == Verify::Homography) {
        cv::Mat H = cv::findHomography(pa, pb, cv::RANSAC, thresh, cv_mask);
        if (H.empty()) return mask;
    } else {
        cv::Mat F = cv::findFundamentalMat(pa, pb, cv::FM_RANSAC, thresh, 0.99, cv_mask);
        if (F.empty()) return mask;
    }
    if (cv_mask.rows != (int)pa.size()) return mask;

    out_ran = true;
    for (int i = 0; i < cv_mask.rows; ++i) mask[i] = cv_mask.at<uchar>(i, 0);
    return mask;
}

cv::Mat toColor(const cv::Mat& gray) {
    cv::Mat color;
    if (gray.channels() == 1) cv::cvtColor(gray, color, cv::COLOR_GRAY2BGR);
    else                      color = gray.clone();
    return color;
}

// Composes the annotated pair image. Frame A sits left (or top), frame B right
// (or bottom); every match becomes one line across the seam.
cv::Mat renderPair(const cv::Mat& img_a, const cv::Mat& img_b,
                   const std::vector<cv::KeyPoint>& kp_a,
                   const std::vector<cv::KeyPoint>& kp_b,
                   const std::vector<cv::DMatch>& matches,
                   const std::vector<uchar>& inlier_mask,
                   bool verified,
                   const Options& opt,
                   const std::string& header) {
    const cv::Mat ca = toColor(img_a);
    const cv::Mat cb = toColor(img_b);

    const int band = 46;   // header strip height
    cv::Point2f off_b;     // where frame B's origin lands on the canvas
    cv::Mat canvas;
    if (opt.layout == Layout::SideBySide) {
        canvas = cv::Mat::zeros(band + std::max(ca.rows, cb.rows), ca.cols + cb.cols, CV_8UC3);
        off_b = cv::Point2f((float)ca.cols, (float)band);
    } else {
        canvas = cv::Mat::zeros(band + ca.rows + cb.rows, std::max(ca.cols, cb.cols), CV_8UC3);
        off_b = cv::Point2f(0.0f, (float)(band + ca.rows));
    }
    const cv::Point2f off_a(0.0f, (float)band);

    ca.copyTo(canvas(cv::Rect((int)off_a.x, (int)off_a.y, ca.cols, ca.rows)));
    cb.copyTo(canvas(cv::Rect((int)off_b.x, (int)off_b.y, cb.cols, cb.rows)));

    // Unmatched keypoints first, so match lines draw over them.
    if (opt.draw_unmatched) {
        std::vector<char> used_a(kp_a.size(), 0), used_b(kp_b.size(), 0);
        for (const auto& m : matches) { used_a[m.queryIdx] = 1; used_b[m.trainIdx] = 1; }
        const cv::Scalar dim(120, 120, 120);
        for (size_t i = 0; i < kp_a.size(); ++i)
            if (!used_a[i]) cv::circle(canvas, kp_a[i].pt + off_a, 1, dim, -1, cv::LINE_AA);
        for (size_t i = 0; i < kp_b.size(); ++i)
            if (!used_b[i]) cv::circle(canvas, kp_b[i].pt + off_b, 1, dim, -1, cv::LINE_AA);
    }

    // Match lines. Outliers go down first so a dense inlier field stays readable.
    const cv::Scalar inlier_col(80, 220, 80);     // green
    const cv::Scalar outlier_col(60, 60, 235);    // red
    const cv::Scalar plain_col(70, 200, 235);     // amber, when unverified
    // --max-lines subsamples with an even stride rather than truncating: the
    // first N matches all come from one corner of the descriptor ordering, and
    // a thinned-but-spread field is what makes a dense pair readable.
    std::vector<int> draw_idx;
    if (opt.max_lines > 0 && (int)matches.size() > opt.max_lines) {
        draw_idx.reserve(opt.max_lines);
        for (int k = 0; k < opt.max_lines; ++k) {
            draw_idx.push_back((int)((size_t)k * matches.size() / opt.max_lines));
        }
    } else {
        draw_idx.resize(matches.size());
        for (size_t k = 0; k < matches.size(); ++k) draw_idx[k] = (int)k;
    }

    for (int pass = 0; pass < 2; ++pass) {
        for (int i : draw_idx) {
            const bool inlier = inlier_mask[i] != 0;
            if ((pass == 0) == inlier) continue;   // pass 0 = outliers, pass 1 = inliers

            const cv::Point2f a = kp_a[matches[i].queryIdx].pt + off_a;
            const cv::Point2f b = kp_b[matches[i].trainIdx].pt + off_b;
            const cv::Scalar col = !verified ? plain_col : (inlier ? inlier_col : outlier_col);
            cv::line(canvas, a, b, col, 1, cv::LINE_AA);
            cv::circle(canvas, a, 3, col, 1, cv::LINE_AA);
            cv::circle(canvas, b, 3, col, 1, cv::LINE_AA);
        }
    }

    cv::rectangle(canvas, cv::Rect(0, 0, canvas.cols, band), cv::Scalar(25, 25, 25), -1);
    // Shrink the header font until it fits the canvas width: the stacked layout
    // is only one frame wide, and a clipped stats line defeats the purpose.
    double font_scale = 0.6;
    int baseline = 0;
    while (font_scale > 0.3 &&
           cv::getTextSize(header, cv::FONT_HERSHEY_SIMPLEX, font_scale, 1, &baseline).width
               > canvas.cols - 20) {
        font_scale -= 0.05;
    }
    cv::putText(canvas, header, cv::Point(10, 29), cv::FONT_HERSHEY_SIMPLEX, font_scale,
                cv::Scalar(240, 240, 240), 1, cv::LINE_AA);
    return canvas;
}

}  // namespace

int main(int argc, char** argv) {
    setvbuf(stdout, NULL, _IONBF, 0);
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_SILENT);

    Options opt;
    if (!parseArgs(argc, argv, opt)) { printUsage(); return -1; }

    // Config: everything is optional except that we end up with an image folder.
    OdometryConfig config;
    config.backend = ComputeBackend::CPU;

    std::unique_ptr<ICameraModel> camera;
    if (!opt.yaml_file.empty()) {
        ConfigLoader loader(opt.yaml_file);
        if (!loader.isOpen()) {
            std::cerr << "Failed to open YAML file. Did you add '%YAML:1.0' to the top of the file?\n";
            return -1;
        }
        loader.loadOdometryConfig(config);

        ImageSequenceConfig seq_cfg;
        if (opt.image_dir.empty() && loader.loadImageSequenceConfig(seq_cfg)) {
            opt.image_dir = seq_cfg.path;
        }

        CameraModelConfig cam_cfg;
        if (opt.undistort && loader.loadCameraModel(cam_cfg)) {
            camera = CameraModelFactory::create(cam_cfg);
            if (camera) {
                config.intrinsics = camera->intrinsics();
                const cv::Size sz = camera->outputSize();
                std::cout << "[Camera] " << cam_cfg.model_type << " -> rectified "
                          << sz.width << "x" << sz.height << "\n";
            }
        }
    }

    if (!opt.detector_override.empty()) config.detector_type = opt.detector_override;
    if (!opt.matcher_override.empty())  config.matcher_type  = opt.matcher_override;
    if (opt.debug) config.verbose = true;

    if (opt.image_dir.empty()) {
        std::cerr << "No image folder: pass --images <dir> or a YAML with image_sequence.path.\n";
        printUsage();
        return -1;
    }
    if (!std::filesystem::exists(opt.image_dir)) {
        std::cerr << "Image folder does not exist: " << opt.image_dir << "\n";
        return -1;
    }

#ifdef USE_ONNX
    if (config.backend == ComputeBackend::CUDA) {
        CudaPreload::init(config.verbose);
    }
#endif

    // Pair list: explicit --pairs, else the --start/--count/--gap/--stride walk.
    std::vector<std::pair<int, int>> pairs = opt.pairs;
    if (pairs.empty()) {
        for (int i = 0; i < opt.count; ++i) {
            const int a = opt.start + i * opt.stride;
            pairs.emplace_back(a, a + opt.gap);
        }
    }

    std::filesystem::create_directories(opt.out_dir);

    auto detector = DetectorFactory::create(config);
    auto matcher  = MatcherFactory::create(config);

    std::cout << "[MatchDebug] " << config.detector_type << " + " << config.matcher_type
              << " on " << opt.image_dir << " (" << pairs.size() << " pairs)\n";

    int written = 0;
    for (const auto& [id_a, id_b] : pairs) {
        const std::string path_a = framePath(opt.image_dir, id_a);
        const std::string path_b = framePath(opt.image_dir, id_b);

        cv::Mat img_a = cv::imread(path_a, cv::IMREAD_GRAYSCALE);
        cv::Mat img_b = cv::imread(path_b, cv::IMREAD_GRAYSCALE);
        if (img_a.empty() || img_b.empty()) {
            std::cerr << "[MatchDebug] Skipping pair " << id_a << ":" << id_b
                      << " — could not read " << (img_a.empty() ? path_a : path_b) << "\n";
            continue;
        }
        if (camera) {
            img_a = camera->undistortImage(img_a);
            img_b = camera->undistortImage(img_b);
        }

        DeviceBuffer buf_a(img_a), buf_b(img_b);
        std::vector<cv::KeyPoint> kp_a, kp_b;
        DeviceBuffer desc_a, desc_b;

        PairStats st;
        auto t0 = std::chrono::high_resolution_clock::now();
        detector->detect(buf_a, kp_a, desc_a);
        detector->detect(buf_b, kp_b, desc_b);
        auto t1 = std::chrono::high_resolution_clock::now();
        st.detect_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        t0 = std::chrono::high_resolution_clock::now();
        std::vector<cv::DMatch> matches = matcher->match(desc_a, desc_b, kp_a, kp_b);
        t1 = std::chrono::high_resolution_clock::now();
        st.match_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        st.kp_a = (int)kp_a.size();
        st.kp_b = (int)kp_b.size();
        st.matches = (int)matches.size();

        std::vector<cv::Point2f> pa, pb;
        pa.reserve(matches.size());
        pb.reserve(matches.size());
        std::vector<double> disp;
        disp.reserve(matches.size());
        for (const auto& m : matches) {
            pa.push_back(kp_a[m.queryIdx].pt);
            pb.push_back(kp_b[m.trainIdx].pt);
            disp.push_back(cv::norm(pb.back() - pa.back()));
        }
        if (!disp.empty()) {
            std::nth_element(disp.begin(), disp.begin() + disp.size() / 2, disp.end());
            st.median_disp = disp[disp.size() / 2];
        }

        bool verified = false;
        std::vector<uchar> mask = verifyMatches(pa, pb, opt.verify, opt.ransac_thresh, verified);
        st.inliers = 0;
        for (uchar v : mask) st.inliers += (v != 0);

        char header[512];
        const char* verify_name = opt.verify == Verify::None ? "none"
                                : opt.verify == Verify::Homography ? "H" : "F";
        if (verified) {
            snprintf(header, sizeof(header),
                     "%s+%s  f%06d->f%06d  kpts %d/%d  matches %d  inliers %d (%.0f%%, %s)  "
                     "med disp %.1fpx  det %.0fms match %.0fms",
                     config.detector_type.c_str(), config.matcher_type.c_str(), id_a, id_b,
                     st.kp_a, st.kp_b, st.matches, st.inliers,
                     st.matches ? 100.0 * st.inliers / st.matches : 0.0, verify_name,
                     st.median_disp, st.detect_ms, st.match_ms);
        } else {
            snprintf(header, sizeof(header),
                     "%s+%s  f%06d->f%06d  kpts %d/%d  matches %d  (unverified)  "
                     "med disp %.1fpx  det %.0fms match %.0fms",
                     config.detector_type.c_str(), config.matcher_type.c_str(), id_a, id_b,
                     st.kp_a, st.kp_b, st.matches, st.median_disp, st.detect_ms, st.match_ms);
        }

        cv::Mat canvas = renderPair(img_a, img_b, kp_a, kp_b, matches, mask, verified, opt, header);

        char out_name[1024];
        snprintf(out_name, sizeof(out_name), "%s/match_%06d_%06d.png",
                 opt.out_dir.c_str(), id_a, id_b);
        if (!cv::imwrite(out_name, canvas)) {
            std::cerr << "[MatchDebug] Failed to write " << out_name << "\n";
            continue;
        }
        ++written;
        std::cout << "[MatchDebug] " << header << "  -> " << out_name << "\n";
    }

    std::cout << "[MatchDebug] Wrote " << written << "/" << pairs.size()
              << " debug frames to " << opt.out_dir << "\n";
    return written > 0 ? 0 : -1;
}
