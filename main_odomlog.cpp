#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <filesystem>
#include <algorithm>
#include <cmath>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/utils/logger.hpp>

#include "odometry/OdometryPipeline.h"
#include "odometry/OdometryTypes.h"
#include "odometry/utils/ConfigLoader.h"
#include "odometry/camera/CameraModelConfig.h"
#include "odometry/camera/CameraModelFactory.h"
#include "odometry/camera/ICameraModel.h"
#ifdef USE_ONNX
#include "odometry/utils/CudaPreload.h"
#endif
#include "odometry/utils/PoseMath.h"
#include "odometry/utils/RealTime2DTrajectory.h"

namespace {

struct TimedFrame {
    int frame_id;
    double timestamp;
};

// Parses a "filename,timestamp" CSV (header row skipped). Malformed or
// non-matching lines are dropped rather than aborting the whole file.
std::vector<TimedFrame> loadTimesFile(const std::string& path) {
    std::vector<TimedFrame> out;
    std::ifstream f(path);
    if (!f.is_open()) return out;

    std::string line;
    bool first_line = true;
    while (std::getline(f, line)) {
        if (first_line) { first_line = false; continue; }   // header
        if (line.empty()) continue;

        size_t comma = line.find(',');
        if (comma == std::string::npos) continue;

        int frame_id;
        if (sscanf(line.substr(0, comma).c_str(), "frame_%d.jpg", &frame_id) != 1) continue;

        double ts;
        try {
            ts = std::stod(line.substr(comma + 1));
        } catch (...) {
            continue;
        }
        out.push_back({frame_id, ts});
    }
    return out;
}

// Full-folder fallback when no times_file is configured: every contiguous
// frame_NNNNNN.jpg starting at 0.
std::vector<TimedFrame> scanFolder(const std::string& dir) {
    std::vector<TimedFrame> out;
    for (int frame_id = 0; ; ++frame_id) {
        char path[512];
        snprintf(path, sizeof(path), "%s/frame_%06d.jpg", dir.c_str(), frame_id);
        if (!std::filesystem::exists(path)) break;
        // No timestamps without a times_file; frame_id doubles as the key.
        out.push_back({frame_id, static_cast<double>(frame_id)});
    }
    return out;
}

// One MAVLink GLOBAL_POSITION_INT sample, already reduced to a local-tangent-
// plane ENU position (metres) relative to the first valid fix.
struct GTSample {
    double timestamp;      // seconds (absolute, matches the times_file clock)
    cv::Vec3f position;    // X=East, Y=North, Z=Up
    // GLOBAL_POSITION_INT.relative_alt: height above the *home* point, not above
    // the terrain. Feeds the homography altimeter scale when altimeter_scale is
    // set. Only equals true AGL where the ground sits at takeoff elevation --
    // terrain relief maps straight into a systematic scale error. Untouched by
    // the caller's ENU re-origin, which shifts `position` only.
    float altitude;
};

// Local-tangent-plane ENU offset (X=East, Y=North, Z=Up) from an origin fix.
// Equirectangular approximation; fine over the short baselines flown here.
cv::Vec3f enuOffset(double origin_lat, double origin_lon, double origin_alt,
                    double lat, double lon, double alt) {
    constexpr double kEarthRadius = 6378137.0;   // WGS84 equatorial radius (m)
    const double deg = M_PI / 180.0;
    double d_east  = kEarthRadius * (lon - origin_lon) * deg * std::cos(origin_lat * deg);
    double d_north = kEarthRadius * (lat - origin_lat) * deg;
    double d_up    = alt - origin_alt;
    return cv::Vec3f((float)d_east, (float)d_north, (float)d_up);
}

// Parses a MAVLink GLOBAL_POSITION_INT_log.csv:
//   timestamp,time_boot_ms,lat,lon,alt,relative_alt
// lat/lon are degrees x 1e7, alt and relative_alt are mm (MSL / above-home).
// Rows before GPS lock log lat==lon==0 and are dropped. The first surviving fix
// is a provisional ENU origin; the caller re-origins the track at the first
// processed frame.
std::vector<GTSample> loadGroundTruthFile(const std::string& path) {
    std::vector<GTSample> out;
    std::ifstream f(path);
    if (!f.is_open()) return out;

    std::string line;
    bool first_line = true;
    bool have_origin = false;
    double olat = 0.0, olon = 0.0, oalt = 0.0;
    while (std::getline(f, line)) {
        if (first_line) { first_line = false; continue; }   // header
        if (line.empty()) continue;

        double ts;
        long long tboot, lat_e7, lon_e7, alt_mm, rel_mm;
        if (sscanf(line.c_str(), "%lf,%lld,%lld,%lld,%lld,%lld",
                   &ts, &tboot, &lat_e7, &lon_e7, &alt_mm, &rel_mm) < 5) continue;
        if (lat_e7 == 0 && lon_e7 == 0) continue;            // no GPS fix yet

        double lat = lat_e7 / 1e7;
        double lon = lon_e7 / 1e7;
        double alt = alt_mm / 1000.0;
        if (!have_origin) { olat = lat; olon = lon; oalt = alt; have_origin = true; }
        out.push_back({ts, enuOffset(olat, olon, oalt, lat, lon, alt),
                       static_cast<float>(rel_mm / 1000.0)});
    }
    return out;
}

// Linearly interpolated lookup into a time-sorted GT track. Clamps to the ends
// outside the sample window rather than extrapolating.
//
// Interpolating (rather than snapping to the nearest fix) matters whenever the
// GPS log is slower than the camera -- run6 logs GLOBAL_POSITION_INT at 5 Hz
// against 10 Hz frames, so nearest-neighbour hands consecutive frames an
// identical position and ~half the keyframes see a zero GT delta, i.e. zero
// metric scale. Same treatment main_rosbag gives its PPK track.
// Position and altitude are interpolated together; the returned sample's
// timestamp is the query time.
GTSample groundTruthAt(const std::vector<GTSample>& gt, double ts) {
    auto it = std::lower_bound(gt.begin(), gt.end(), ts,
                               [](const GTSample& s, double t) { return s.timestamp < t; });
    if (it == gt.begin()) return gt.front();
    if (it == gt.end())   return gt.back();
    const GTSample& hi = *it;
    const GTSample& lo = *(it - 1);
    const double span = hi.timestamp - lo.timestamp;
    if (span <= 0.0) return lo;
    const float a = static_cast<float>((ts - lo.timestamp) / span);
    return {ts,
            lo.position + (hi.position - lo.position) * a,
            lo.altitude + (hi.altitude - lo.altitude) * a};
}

// One MAVLink ATTITUDE sample, reduced to the only field the heading seed uses.
struct HeadingSample {
    double timestamp;   // seconds, on the same clock as times_file
    double yaw;         // NED radians (0 = North, +clockwise)
};

// Parses a MAVLink ATTITUDE_log.csv:
//   timestamp,time_boot_ms,roll,pitch,yaw
// roll/pitch/yaw are radians in the NED body frame. Short or malformed lines
// are dropped rather than aborting -- some captures (run6) end in a partially
// flushed record padded with NULs, which sscanf rejects on its own.
std::vector<HeadingSample> loadHeadingFile(const std::string& path) {
    std::vector<HeadingSample> out;
    std::ifstream f(path);
    if (!f.is_open()) return out;

    std::string line;
    bool first_line = true;
    while (std::getline(f, line)) {
        if (first_line) { first_line = false; continue; }   // header
        if (line.empty()) continue;

        double ts, roll, pitch, yaw;
        long long tboot;
        if (sscanf(line.c_str(), "%lf,%lld,%lf,%lf,%lf",
                   &ts, &tboot, &roll, &pitch, &yaw) != 5) continue;
        out.push_back({ts, yaw});
    }
    std::sort(out.begin(), out.end(),
              [](const HeadingSample& a, const HeadingSample& b) {
                  return a.timestamp < b.timestamp;
              });
    return out;
}

// Shortest-path interpolation between two angles in radians, so a query that
// straddles the +-pi wrap does not sweep the long way round.
double lerpAngleRad(double a, double b, double f) {
    double d = std::fmod(b - a + 3.0 * M_PI, 2.0 * M_PI) - M_PI;
    return a + f * d;
}

// Interpolated heading lookup. Clamps to the ends outside the sample window,
// matching groundTruthAt's behaviour. Returns false only for an empty track.
bool headingAt(const std::vector<HeadingSample>& track, double ts, double& out_yaw) {
    if (track.empty()) return false;
    auto it = std::lower_bound(track.begin(), track.end(), ts,
                               [](const HeadingSample& s, double t) { return s.timestamp < t; });
    if (it == track.begin()) { out_yaw = track.front().yaw; return true; }
    if (it == track.end())   { out_yaw = track.back().yaw;  return true; }
    const HeadingSample& hi = *it;
    const HeadingSample& lo = *(it - 1);
    const double span = hi.timestamp - lo.timestamp;
    out_yaw = (span <= 0.0) ? lo.yaw
                            : lerpAngleRad(lo.yaw, hi.yaw, (ts - lo.timestamp) / span);
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: ./OdomLogEvaluator <yaml_config_path> [--debug] [--no-gui] [--out <csv>]\n";
        return -1;
    }

    setvbuf(stdout, NULL, _IONBF, 0);
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_SILENT);

    std::string yaml_file;
    std::string out_path = "odomlog_trajectory.csv";
    bool cli_debug = false;
    bool no_gui = false;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--debug") cli_debug = true;
        else if (arg == "--no-gui") no_gui = true;
        else if (arg == "--out" && i + 1 < argc) out_path = argv[++i];
        else if (yaml_file.empty()) yaml_file = arg;
    }
    if (yaml_file.empty()) {
        std::cerr << "Usage: ./OdomLogEvaluator <yaml_config_path> [--debug] [--no-gui] [--out <csv>]\n";
        return -1;
    }

    OdometryConfig config;
    config.backend = ComputeBackend::CPU;

    ConfigLoader loader(yaml_file);
    if (!loader.isOpen()) {
        std::cerr << "Failed to open YAML file. Did you add '%YAML:1.0' to the top of the file?\n";
        return -1;
    }
    loader.loadOdometryConfig(config);
    if (cli_debug) config.verbose = true;   // --debug overrides system.verbose.

#ifdef USE_ONNX
    if (config.backend == ComputeBackend::CUDA) {
        CudaPreload::init(config.verbose);
    }
#endif

    ImageSequenceConfig seq_cfg;
    if (!loader.loadImageSequenceConfig(seq_cfg)) {
        std::cerr << "YAML is missing the image_sequence.path block.\n";
        return -1;
    }

    std::vector<TimedFrame> frames;
    if (!seq_cfg.times_file.empty()) {
        std::vector<TimedFrame> entries = loadTimesFile(seq_cfg.times_file);
        if (entries.empty()) {
            std::cerr << "[EVALUATOR] Warning: could not parse times_file '" << seq_cfg.times_file
                       << "'; ignoring start_time/end_time and scanning the full folder.\n";
        } else {
            double t0 = entries.front().timestamp;
            for (const TimedFrame& e : entries) {
                double elapsed = e.timestamp - t0;
                if (elapsed < seq_cfg.start_time) continue;
                if (seq_cfg.end_time >= 0.0 && elapsed > seq_cfg.end_time) continue;
                frames.push_back(e);
            }
        }
    }
    if (frames.empty() && seq_cfg.times_file.empty()) {
        frames = scanFolder(seq_cfg.path);
    }

    // Optional ground truth (MAVLink GLOBAL_POSITION_INT). Needs the times_file
    // clock to align GPS fixes with the frames, so it's only used when both are
    // present. When loaded, each frame gets a nearest-in-time metric ENU
    // position; combined with scale_estimator.type: AirSim this yields a metric
    // trajectory instead of the unitless shape-check.
    std::vector<GTSample> gt_track;
    if (!seq_cfg.ground_truth_file.empty()) {
        if (seq_cfg.times_file.empty()) {
            std::cerr << "[EVALUATOR] Warning: ground_truth_file set but no times_file; "
                         "cannot align GT to frames, ignoring it.\n";
        } else {
            gt_track = loadGroundTruthFile(seq_cfg.ground_truth_file);
            if (gt_track.empty()) {
                std::cerr << "[EVALUATOR] Warning: could not parse ground_truth_file '"
                           << seq_cfg.ground_truth_file << "' (or it holds no GPS fix); "
                              "running without ground truth.\n";
            }
        }
    }
    const bool have_gt = !gt_track.empty();

    // Optional deployment-mode heading seed (MAVLink ATTITUDE). Replaces the GT
    // auto-aligner: the yaw is solved once at the first tracked frame from the
    // heading sensor alone, so nothing about the alignment depends on ground
    // truth. Needs the times_file clock for the same reason the GT track does.
    std::vector<HeadingSample> heading_track;
    if (!seq_cfg.heading_file.empty()) {
        if (seq_cfg.times_file.empty()) {
            std::cerr << "[EVALUATOR] Warning: heading_source.path set but no times_file; "
                         "cannot align headings to frames, ignoring it.\n";
        } else {
            heading_track = loadHeadingFile(seq_cfg.heading_file);
            if (heading_track.empty()) {
                std::cerr << "[EVALUATOR] Warning: could not parse heading_source.path '"
                           << seq_cfg.heading_file << "'; falling back to the GT aligner.\n";
            }
        }
    }
    const bool use_heading_seed = !heading_track.empty();

    // Re-origin the ENU track at the first processed frame so the logged GT
    // starts near zero rather than at the drone's position when the log's first
    // GPS fix landed. Scale is delta-based, so this shifts only the GT columns,
    // not the trajectory.
    if (have_gt && !frames.empty()) {
        cv::Vec3f gt_origin = groundTruthAt(gt_track, frames.front().timestamp).position;
        for (GTSample& s : gt_track) s.position -= gt_origin;
    }

    // Camera model (optional). When camera.model_type is present the model owns
    // undistortion and supplies the rectified pinhole intrinsics the pipeline
    // runs on; otherwise the plain fx/fy/cx/cy from loadOdometryConfig stand.
    std::unique_ptr<ICameraModel> camera;
    CameraModelConfig cam_cfg;
    if (loader.loadCameraModel(cam_cfg)) {
        camera = CameraModelFactory::create(cam_cfg);
        if (camera) {
            config.intrinsics = camera->intrinsics();
            const cv::Size sz = camera->outputSize();
            std::cout << "[Camera] " << cam_cfg.model_type << " -> rectified "
                      << sz.width << "x" << sz.height
                      << " (fx=" << config.intrinsics.fx
                      << " fy=" << config.intrinsics.fy
                      << " cx=" << config.intrinsics.cx
                      << " cy=" << config.intrinsics.cy << ")\n";
        }
    }

    auto pipeline = OdometryPipeline::build(config);

    std::ofstream log_file(out_path);
    log_file << "Frame,Pred_X,Pred_Y,Pred_Z,Q_X,Q_Y,Q_Z,Q_W,GT_X,GT_Y,GT_Z\n";

    std::cout << "[EVALUATOR] Starting odometry_log replay: " << seq_cfg.path << " ("
               << frames.size() << " frames)\n";
    if (have_gt) {
        std::cout << "[EVALUATOR] Ground truth loaded from " << seq_cfg.ground_truth_file << " ("
                   << gt_track.size() << " GPS fixes). Per-frame metric ENU position feeds the\n"
                     "            scale estimator; with scale_estimator.type: AirSim the trajectory\n"
                     "            is metric. GT columns are logged (ENU, origin at first frame). The\n"
                     "            plot overlays estimate (XY/East-North) vs GT.\n";
        if (!use_heading_seed) {
            std::cout << "[EVALUATOR] Yaw alignment: GT auto-aligner, solved after "
                      << seq_cfg.aligner_init_distance << "m of GT travel.\n";
        }
        if (seq_cfg.altimeter_scale) {
            std::cout << "[EVALUATOR] Altimeter scale ON: relative_alt (above home, NOT terrain)\n"
                         "            feeds the Homography plane distance. Frames with AGL <= 0\n"
                         "            fall back to the GT-delta scale.\n";
        }
    } else {
        std::cout << "[EVALUATOR] No ground truth available for this dataset. The 2D trajectory\n"
                     "            plot and CSV log both show the VO estimate only (fed to itself\n"
                     "            wherever a GT value would normally go) -- it's a shape check, not\n"
                     "            an accuracy measurement.\n";
    }
    if (use_heading_seed) {
        std::cout << "[EVALUATOR] Yaw alignment: DEPLOYMENT MODE -- seeded at the first tracked\n"
                     "            frame from " << heading_track.size() << " ATTITUDE samples in "
                  << seq_cfg.heading_file << ",\n            mount offset "
                  << seq_cfg.heading_mount_offset << " deg. The GT auto-aligner is bypassed, so\n"
                     "            no ground truth enters the trajectory's orientation.\n";
    }
    std::cout << "------------------------------------------------------\n";

    // Fallback GT when the dataset has none: a default-constructed
    // GroundTruthData every frame. With scale_estimator.type: Unit the scale
    // estimator ignores this and returns a fixed unitless step instead of a
    // metric GT-delta, so the trajectory still has a visible shape.
    GroundTruthData no_gt;

    // With GT the plot is ENU East/North (XY) and the estimate is aligned to it;
    // without GT it's the legacy est-only shape check in the XZ plane.
    RealTime2DTrajectory trajectory_visualizer(
        (float)seq_cfg.viz_scale,
        have_gt ? RealTime2DTrajectory::Plane::XY : RealTime2DTrajectory::Plane::XZ);

    // Camera->body rotation, applied on the LEFT of the VO pose to match the
    // Python TrajectoryIntegrator (cur_R = R_cb * R_vo, cur_t = R_cb * t_vo):
    // it rotates the whole VO-world frame into the body frame. The rotation is
    // the 3x3 block of body_T_cam0 (camera->body); the lever arm is not modeled.
    // Identity when no extrinsic is given.
    cv::Mat cam_to_body = cv::Mat::eye(4, 4, CV_64F);
    if (!seq_cfg.body_T_cam0.empty()) {
        cv::Mat ext;
        seq_cfg.body_T_cam0.convertTo(ext, CV_64F);
        ext(cv::Rect(0, 0, 3, 3)).copyTo(cam_to_body(cv::Rect(0, 0, 3, 3)));
        std::cout << "[Extrinsic] Applying body_T_cam0 (camera->body rotation, left-multiply).\n";
    }

    // Per-axis sign flip S = diag(sx,sy,sz,1) for a VO-vs-ENU handedness
    // mismatch (a reflection a yaw can't fix -- e.g. a nadir camera's image-down
    // axis). Applied to the VO pose by conjugation S*T*S, which negates the
    // chosen position axes while keeping the rotation a valid (det +1) matrix.
    cv::Mat traj_flip = cv::Mat::eye(4, 4, CV_64F);
    traj_flip.at<double>(0, 0) = seq_cfg.traj_sign[0];
    traj_flip.at<double>(1, 1) = seq_cfg.traj_sign[1];
    traj_flip.at<double>(2, 2) = seq_cfg.traj_sign[2];
    if (seq_cfg.traj_sign != cv::Vec3d(1.0, 1.0, 1.0)) {
        std::cout << "[Trajectory] Axis sign flip: (" << seq_cfg.traj_sign[0] << ", "
                  << seq_cfg.traj_sign[1] << ", " << seq_cfg.traj_sign[2] << ")\n";
    }

    // XY-plane auto-aligner (only meaningful with GT): once GT has travelled
    // aligner_init_distance from the first tracked frame, solve the yaw between
    // the VO and GT displacement vectors and rotate the VO trajectory about Z by
    // it. Identity until then. Same construction as main_rosbag.
    cv::Mat align_T_vo = cv::Mat::eye(4, 4, CV_64F);
    bool align_ready = false;
    bool align_start_set = false;
    cv::Point2d vo_start, gt_start;

    for (const TimedFrame& tf : frames) {
        int frame_id = tf.frame_id;
        char img_name[512];
        snprintf(img_name, sizeof(img_name), "%s/frame_%06d.jpg", seq_cfg.path.c_str(), frame_id);

        cv::Mat frame = cv::imread(img_name, cv::IMREAD_GRAYSCALE);
        if (frame.empty()) continue;

        if (camera) frame = camera->undistortImage(frame);

        DeviceBuffer frame_buffer(frame);

        // Interpolate GT at this frame's timestamp when available; otherwise the
        // default (zero) GT, which only the Unit scale estimator makes use of.
        GroundTruthData current_gt = no_gt;
        if (have_gt) {
            const GTSample s = groundTruthAt(gt_track, tf.timestamp);
            current_gt.position = s.position;
            // Altimeter scale (Homography only). Left at the -1 default when
            // disabled, and skipped for a non-positive AGL (on the ground /
            // pre-takeoff), where the pipeline falls back to the GT-delta scale.
            if (seq_cfg.altimeter_scale && s.altitude > 0.0f) current_gt.altitude = s.altitude;
        }

        pipeline->processFrame(frame_buffer, current_gt);

        if (pipeline->isTrackingActive()) {
            // VO pose in the display frame: camera->body rotation left-applied
            // (Python-style, rotates the trajectory into the body frame) and the
            // handedness flip by conjugation, but not yet the yaw alignment.
            cv::Mat M = traj_flip * cam_to_body
                        * pipeline->getGlobalTransformVO() * traj_flip;

            // Deployment-mode heading seed. Solves the same Z rotation the GT
            // aligner does, but from the heading sensor at this first tracked
            // frame: align_yaw = mount_offset - heading(t0). Setting
            // align_ready here means the GT aligner below never runs, and the
            // trajectory is aligned from frame 0 rather than after the first
            // aligner_init_distance metres.
            if (use_heading_seed && !align_ready) {
                double h_ned = 0.0;
                if (headingAt(heading_track, tf.timestamp, h_ned)) {
                    const double yaw = seq_cfg.heading_mount_offset * M_PI / 180.0 - h_ned;
                    const double c = std::cos(yaw), s = std::sin(yaw);
                    align_T_vo = (cv::Mat_<double>(4, 4) <<
                        c, -s, 0, 0,
                        s,  c, 0, 0,
                        0,  0, 1, 0,
                        0,  0, 0, 1);
                    align_ready = true;
                    std::cout << "\n[Heading] seeded from " << seq_cfg.heading_file
                              << ": heading=" << h_ned * 180.0 / M_PI
                              << " deg (NED) + mount offset "
                              << seq_cfg.heading_mount_offset << " deg -> yaw="
                              << std::remainder(yaw * 180.0 / M_PI, 360.0)
                              << " deg at frame " << frame_id << " (no GT used)\n";
                }
            }

            // Auto-aligner: record the VO and GT start once tracking begins, and
            // once GT has travelled aligner_init_distance, solve the yaw that
            // rotates the VO displacement onto the GT displacement. Held after.
            if (have_gt && !align_ready) {
                cv::Point2d p_vo(M.at<double>(0, 3), M.at<double>(1, 3));
                cv::Point2d p_gt(current_gt.position[0], current_gt.position[1]);
                if (!align_start_set) {
                    vo_start = p_vo;
                    gt_start = p_gt;
                    align_start_set = true;
                }
                cv::Point2d gt_disp = p_gt - gt_start;
                if (cv::norm(gt_disp) >= seq_cfg.aligner_init_distance) {
                    cv::Point2d vo_disp = p_vo - vo_start;
                    if (cv::norm(vo_disp) > 1e-6) {
                        const double yaw = std::atan2(gt_disp.y, gt_disp.x)
                                         - std::atan2(vo_disp.y, vo_disp.x);
                        const double c = std::cos(yaw), s = std::sin(yaw);
                        align_T_vo = (cv::Mat_<double>(4, 4) <<
                            c, -s, 0, 0,
                            s,  c, 0, 0,
                            0,  0, 1, 0,
                            0,  0, 0, 1);
                        align_ready = true;
                        std::cout << "\n[Aligner] yaw=" << yaw * 180.0 / M_PI
                                  << " deg after " << cv::norm(gt_disp) << " m of GT travel\n";
                    }
                }
            }

            // Apply the yaw alignment outermost so the logged pose and plot land
            // in the GT ENU frame (identity when no GT / not yet aligned).
            cv::Mat T_world = align_T_vo * M;
            float pred_x = T_world.at<double>(0, 3);
            float pred_y = T_world.at<double>(1, 3);
            float pred_z = T_world.at<double>(2, 3);

            cv::Mat R = T_world(cv::Rect(0, 0, 3, 3));
            float qx, qy, qz, qw;
            PoseMath::rot2quat(R, qx, qy, qz, qw);

            const cv::Vec3f& gt_xyz = current_gt.position;
            log_file << frame_id << "," << pred_x << "," << pred_y << "," << pred_z << ","
                     << qx << "," << qy << "," << qz << "," << qw << ","
                     << gt_xyz[0] << "," << gt_xyz[1] << "," << gt_xyz[2] << "\n";

            // Plot estimate vs GT. Without GT there's nothing to align to, so the
            // estimate is fed as both args (legacy shape check).
            cv::Vec3f est_xyz(pred_x, pred_y, pred_z);
            cv::Mat traj_img = trajectory_visualizer.update(
                est_xyz, have_gt ? current_gt.position : est_xyz);
            if (!no_gui) cv::imshow("OdomLog 2D Trajectory", traj_img);
        }

        const PipelineMetrics& m = pipeline->getMetrics();
        printf("\r[%05d] Det:%4.0fms | Mat:%4.0fms | Pos:%3.0fms | Tot:%4.0fms | FPS:%4.1f   ",
               frame_id,
               m.time_detect_ms,
               m.time_match_ms,
               m.time_pose_ms,
               m.time_total_ms,
               m.fps);
        fflush(stdout);

        if (!no_gui) {
            cv::Mat vis = pipeline->getDebugFrame();
            if (!vis.empty()) {
                cv::imshow("OdomLog Feature Tracking", vis);
                if (cv::waitKey(1) == 27) break;
            }
        }
    }

    printf("\n");

    log_file.close();
    std::cout << "\n[EVALUATOR] Complete. Trajectory saved to " << out_path << "\n";
    return 0;
}
