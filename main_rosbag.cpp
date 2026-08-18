#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <string>
#include <cmath>
#include <ctime>
#include <algorithm>
#include <limits>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/utils/logger.hpp>

#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/NavSatFix.h>
#include <sensor_msgs/Imu.h>

#include "odometry/OdometryPipeline.h"
#include "odometry/OdometryTypes.h"
#include "odometry/camera/CameraModelFactory.h"
#include "odometry/camera/ICameraModel.h"
#include "odometry/utils/ConfigLoader.h"
#ifdef USE_ONNX
#include "odometry/utils/CudaPreload.h"
#endif
#include "odometry/utils/PoseMath.h"
#include "odometry/utils/RealTime2DTrajectory.h"

// NED roll/pitch/yaw (degrees) to 3x3 rotation matrix (double). Standard
// aerospace body-to-NED sequence R = Rz(yaw) * Ry(pitch) * Rx(roll); this is
// the exact inverse of extractEulerFromRotation, so the round-trip is lossless.
static cv::Mat eulerToRot(double roll_deg, double pitch_deg, double yaw_deg) {
    const double deg = M_PI / 180.0;
    double cr = std::cos(roll_deg * deg),  sr = std::sin(roll_deg * deg);
    double cp = std::cos(pitch_deg * deg), sp = std::sin(pitch_deg * deg);
    double cy = std::cos(yaw_deg * deg),   sy = std::sin(yaw_deg * deg);
    return (cv::Mat_<double>(3,3) <<
        cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr,
        sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr,
          -sp,            cp*sr,            cp*cr);
}

// Right-hand quaternion (x,y,z,w) to 3x3 rotation matrix (double).
static cv::Mat quatToRot(double qx, double qy, double qz, double qw) {
    return (cv::Mat_<double>(3,3) <<
        1 - 2*qy*qy - 2*qz*qz,   2*qx*qy - 2*qz*qw,     2*qx*qz + 2*qy*qw,
        2*qx*qy + 2*qz*qw,       1 - 2*qx*qx - 2*qz*qz, 2*qy*qz - 2*qx*qw,
        2*qx*qz - 2*qy*qw,       2*qy*qz + 2*qx*qw,     1 - 2*qx*qx - 2*qy*qy);
}

// =================== PPK (RTKLib .pos) support ===========================

struct PpkSample {
    int64_t ts_ns;
    double lat, lon, alt;        // WGS84 ellipsoidal (deg, deg, m)
    double roll, pitch, yaw;     // NED frame (deg)
};

// Parse RTKLib / FRL .pos lines. Both position and attitude ground truth come
// from this file. Columns are space-separated:
//   GPST(date time) lat lon height Q ns sdn sde sdu sdne sdeu sdun age ratio
//   roll pitch yaw(deg) P Q R Ve Vn Vu
// lat/lon/height are WGS84 ellipsoidal; roll/pitch/yaw are NED-frame degrees.
// GPS time is converted to UTC by subtracting 18 leap seconds (valid 2017+).
static std::vector<PpkSample> loadPpkFile(const std::string& path) {
    std::vector<PpkSample> out;
    std::ifstream in(path);
    if (!in.is_open()) {
        std::cerr << "[PPK] Failed to open: " << path << "\n";
        return out;
    }

    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '%') continue;

        int year, month, day, hour, minute, Q, ns;
        double sec, lat, lon, alt;
        double sdn, sde, sdu, sdne, sdeu, sdun, age, ratio, roll, pitch, yaw;
        int n = std::sscanf(line.c_str(),
            "%d/%d/%d %d:%d:%lf %lf %lf %lf %d %d %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf",
            &year, &month, &day, &hour, &minute, &sec, &lat, &lon, &alt,
            &Q, &ns, &sdn, &sde, &sdu, &sdne, &sdeu, &sdun, &age, &ratio,
            &roll, &pitch, &yaw);
        if (n != 22) continue;

        std::tm tmv{};
        tmv.tm_year = year - 1900;
        tmv.tm_mon  = month - 1;
        tmv.tm_mday = day;
        tmv.tm_hour = hour;
        tmv.tm_min  = minute;
        tmv.tm_sec  = (int)sec;
        time_t base_utc = timegm(&tmv);
        double frac = sec - (int)sec;

        PpkSample s;
        s.ts_ns = (int64_t)base_utc * 1'000'000'000LL
                + (int64_t)(frac * 1e9)
                - 18LL * 1'000'000'000LL;  // GPST -> UTC
        s.lat = lat; s.lon = lon; s.alt = alt;
        s.roll = roll; s.pitch = pitch; s.yaw = yaw;
        out.push_back(s);
    }

    std::sort(out.begin(), out.end(),
        [](const PpkSample& a, const PpkSample& b){ return a.ts_ns < b.ts_ns; });
    std::cout << "[PPK] Loaded " << out.size() << " samples from " << path << "\n";
    return out;
}

// Shortest-path interpolation between two angles given in degrees.
static double lerpAngleDeg(double a, double b, double f) {
    double d = std::fmod(b - a + 540.0, 360.0) - 180.0;
    return a + f * d;
}

// Linear interpolation of position and attitude; clamps to endpoints outside
// the sample window. Angles use shortest-path interpolation to stay correct
// across the +-180 deg wrap.
static bool interpolatePpk(const std::vector<PpkSample>& ppk, int64_t ts_ns,
                           double& lat, double& lon, double& alt,
                           double& roll, double& pitch, double& yaw) {
    if (ppk.empty()) return false;
    if (ts_ns <= ppk.front().ts_ns) {
        const PpkSample& s = ppk.front();
        lat = s.lat; lon = s.lon; alt = s.alt;
        roll = s.roll; pitch = s.pitch; yaw = s.yaw;
        return true;
    }
    if (ts_ns >= ppk.back().ts_ns) {
        const PpkSample& s = ppk.back();
        lat = s.lat; lon = s.lon; alt = s.alt;
        roll = s.roll; pitch = s.pitch; yaw = s.yaw;
        return true;
    }
    auto it = std::lower_bound(ppk.begin(), ppk.end(), ts_ns,
        [](const PpkSample& s, int64_t t){ return s.ts_ns < t; });
    const PpkSample& hi = *it;
    const PpkSample& lo = *(it - 1);
    double a = double(ts_ns - lo.ts_ns) / double(hi.ts_ns - lo.ts_ns);
    lat = lo.lat + a * (hi.lat - lo.lat);
    lon = lo.lon + a * (hi.lon - lo.lon);
    alt = lo.alt + a * (hi.alt - lo.alt);
    roll  = lerpAngleDeg(lo.roll,  hi.roll,  a);
    pitch = lerpAngleDeg(lo.pitch, hi.pitch, a);
    yaw   = lerpAngleDeg(lo.yaw,   hi.yaw,   a);
    return true;
}

// =================== Bag pre-scan helpers ================================

// Scan the GPS topic once for the lowest reported altitude. Mirrors the
// Python loader's "true ground level" baseline used for AGL.
static double prescanMinGpsAltitude(const std::string& bag_path,
                                    const std::string& gps_topic) {
    std::cout << "[Altimeter] Pre-scanning bag for global minimum GPS altitude...\n";
    rosbag::Bag bag;
    bag.open(bag_path, rosbag::bagmode::Read);
    rosbag::View view(bag, rosbag::TopicQuery(std::vector<std::string>{gps_topic}));

    double min_alt = std::numeric_limits<double>::infinity();
    for (const rosbag::MessageInstance& m : view) {
        auto msg = m.instantiate<sensor_msgs::NavSatFix>();
        if (!msg) continue;
        if (!std::isnan(msg->altitude) && msg->altitude < min_alt) {
            min_alt = msg->altitude;
        }
    }
    bag.close();

    if (std::isinf(min_alt)) {
        std::cout << "[Altimeter] WARNING: No valid GPS altitude found. Defaulting to 0.0\n";
        return 0.0;
    }
    std::cout << "[Altimeter] Found ground level at " << min_alt << "m MSL.\n";
    return min_alt;
}

// =================== ENU helpers =========================================

constexpr double kEarthRadius = 6378137.0;

// Local-tangent-plane ENU offset (X=East, Y=North, Z=Up) from origin.
static cv::Vec3f enuOffset(double origin_lat, double origin_lon, double origin_alt,
                           double lat,        double lon,        double alt) {
    const double deg = M_PI / 180.0;
    double d_east  = kEarthRadius * (lon - origin_lon) * deg * std::cos(origin_lat * deg);
    double d_north = kEarthRadius * (lat - origin_lat) * deg;
    double d_up    = alt - origin_alt;
    return cv::Vec3f((float)d_east, (float)d_north, (float)d_up);
}

// =================== Image decode ========================================

// Returns a single-channel CV_8U cv::Mat. Caller-owned (no aliasing to the
// ROS message buffer). Returns empty Mat on unsupported encoding.
static cv::Mat decodeImage(const sensor_msgs::Image::ConstPtr& msg) {
    if (!msg) return cv::Mat();
    int h = msg->height, w = msg->width;
    if (msg->encoding == "mono8" || msg->encoding == "8UC1") {
        cv::Mat src(h, w, CV_8UC1, (void*)msg->data.data(), msg->step);
        return src.clone();
    }
    if (msg->encoding == "bgr8") {
        cv::Mat src(h, w, CV_8UC3, (void*)msg->data.data(), msg->step);
        cv::Mat gray;
        cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
        return gray;
    }
    if (msg->encoding == "rgb8") {
        cv::Mat src(h, w, CV_8UC3, (void*)msg->data.data(), msg->step);
        cv::Mat gray;
        cv::cvtColor(src, gray, cv::COLOR_RGB2GRAY);
        return gray;
    }
    std::cerr << "[Image] Unsupported encoding: " << msg->encoding << "\n";
    return cv::Mat();
}

// =================== main ================================================

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: ./RosbagEvaluator <yaml_config_path> [--debug]\n";
        return -1;
    }

    setvbuf(stdout, NULL, _IONBF, 0);
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_SILENT);

    std::string yaml_file;
    std::string out_path = "rosbag_trajectory.csv";
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
        std::cerr << "Usage: ./RosbagEvaluator <yaml_config_path> [--debug] [--no-gui] [--out <csv>]\n";
        return -1;
    }

    // OdometryConfig leaves intrinsics zero-init; YAML must supply them.
    OdometryConfig config;
    config.backend = ComputeBackend::CPU;

    ConfigLoader loader(yaml_file);
    if (!loader.isOpen()) {
        std::cerr << "Failed to open YAML file. Did you add '%YAML:1.0' to the top of the file?\n";
        return -1;
    }
    loader.loadOdometryConfig(config);
    if (cli_debug) config.verbose = true;

    RosbagConfig bag_cfg;
    if (!loader.loadRosbagConfig(bag_cfg)) {
        std::cerr << "YAML is missing the rosbag block (rosbag.bag_path).\n";
        return -1;
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

    // cam0->body extrinsic. The logged VO pose (a camera pose in the VO world)
    // is re-expressed at the body via T_world_body = T_world_cam0 * cam0_T_body,
    // where cam0_T_body = (body_T_cam0)^-1. Identity when no extrinsic is given.
    cv::Mat cam0_T_body = cv::Mat::eye(4, 4, CV_64F);
    if (!bag_cfg.body_T_cam0.empty()) {
        cv::Mat ext;
        bag_cfg.body_T_cam0.convertTo(ext, CV_64F);
        cam0_T_body = ext.inv();
        std::cout << "[Extrinsic] Applying body_T_cam0 to logged poses.\n";
    }

    // Per-axis sign flip S = diag(sx,sy,sz,1) for a VO-vs-ENU handedness
    // mismatch. Applied to the logged pose by conjugation S*T*S, which negates
    // the chosen position axes while keeping the rotation a valid (det +1)
    // matrix. Identity when no trajectory_flip block is given.
    cv::Mat traj_flip = cv::Mat::eye(4, 4, CV_64F);
    traj_flip.at<double>(0, 0) = bag_cfg.traj_sign[0];
    traj_flip.at<double>(1, 1) = bag_cfg.traj_sign[1];
    traj_flip.at<double>(2, 2) = bag_cfg.traj_sign[2];
    if (bag_cfg.traj_sign != cv::Vec3d(1.0, 1.0, 1.0)) {
        std::cout << "[Trajectory] Axis sign flip: ("
                  << bag_cfg.traj_sign[0] << ", " << bag_cfg.traj_sign[1]
                  << ", " << bag_cfg.traj_sign[2] << ")\n";
    }

#ifdef USE_ONNX
    if (config.backend == ComputeBackend::CUDA) {
        CudaPreload::init(config.verbose);
    }
#endif

    // PPK overlay (optional). If present, image-timestamped lat/lon/alt come
    // from interpolation here instead of from the GPS topic.
    std::vector<PpkSample> ppk;
    const bool use_ppk = !bag_cfg.ppk_path.empty();
    if (use_ppk) {
        ppk = loadPpkFile(bag_cfg.ppk_path);
        if (ppk.empty()) {
            std::cerr << "[PPK] Aborting: PPK file produced no samples.\n";
            return -1;
        }
    }

    // AGL baseline: PPK min altitude when PPK is active, otherwise pre-scan
    // the bag's GPS topic for the minimum reported altitude.
    double baseline_agl = 0.0;
    if (use_ppk) {
        double m = std::numeric_limits<double>::infinity();
        for (const auto& s : ppk) m = std::min(m, s.alt);
        baseline_agl = std::isinf(m) ? 0.0 : m;
        std::cout << "[Altimeter] PPK ground level: " << baseline_agl << "m MSL.\n";
    } else {
        baseline_agl = prescanMinGpsAltitude(bag_cfg.bag_path, bag_cfg.gps_topic);
    }

    if (bag_cfg.vins_config.empty()) {
        std::cerr << "Fatal: `vins_config:` is missing from " << yaml_file
                  << ". It must point at a VINS-Fusion yaml (camera-IMU extrinsic,\n"
                     "IMU noise densities, td). There is no estimator without it.\n";
        return -1;
    }
    auto pipeline = OdometryPipeline::build(config, bag_cfg.vins_config);

    std::ofstream log_file(out_path);
    log_file << "Frame,Pred_X,Pred_Y,Pred_Z,Q_X,Q_Y,Q_Z,Q_W\n";

    std::cout << "[EVALUATOR] Starting rosbag replay: " << bag_cfg.bag_path << "\n";
    std::cout << "  img_topic=" << bag_cfg.img_topic
              << "  gps_topic=" << bag_cfg.gps_topic
              << "  imu_topic=" << bag_cfg.imu_topic << "\n";
    std::cout << "------------------------------------------------------\n";

    RealTime2DTrajectory trajectory_visualizer((float)bag_cfg.viz_scale,
                                               RealTime2DTrajectory::Plane::XY);

    // Open bag and request the three topics we care about.
    rosbag::Bag bag;
    bag.open(bag_cfg.bag_path, rosbag::bagmode::Read);
    std::vector<std::string> topics = {
        bag_cfg.img_topic, bag_cfg.gps_topic, bag_cfg.imu_topic
    };
    rosbag::View view(bag, rosbag::TopicQuery(topics));

    // Per-frame GT state, populated incrementally from IMU + GPS / PPK.
    bool have_origin = false;
    double origin_lat = 0.0, origin_lon = 0.0, origin_alt = 0.0;
    double cur_alt = 0.0;
    cv::Vec3f cur_position(0, 0, 0);
    cv::Mat   cur_R = cv::Mat::eye(3, 3, CV_64F);

    // XY-plane auto-aligner: once GT has travelled aligner_init_distance from
    // the first tracked frame, solve the yaw between the VO and GT displacement
    // vectors and rotate the VO trajectory about Z by it. Identity until then.
    cv::Mat align_T_vo = cv::Mat::eye(4, 4, CV_64F);
    bool align_ready = false;
    bool align_start_set = false;
    cv::Point2d vo_start, gt_start;

    // Bag-relative time bounds (ns).
    uint64_t bag_start_ns = 0;
    bool bag_start_set = false;
    const int64_t skip_ns = (int64_t)(bag_cfg.start_time * 1e9);
    const bool    has_end = bag_cfg.end_time >= 0.0;
    const int64_t end_ns  = has_end ? (int64_t)(bag_cfg.end_time * 1e9) : 0;

    int frame_id = 0;

    for (const rosbag::MessageInstance& m : view) {
        const std::string& topic = m.getTopic();
        const uint64_t ts_ns = m.getTime().toNSec();

        if (!bag_start_set) {
            bag_start_ns = ts_ns;
            bag_start_set = true;
        }
        const int64_t rel_ns = (int64_t)ts_ns - (int64_t)bag_start_ns;

        if (has_end && rel_ns > end_ns) break;
        if (rel_ns < skip_ns) continue;

        if (topic == bag_cfg.imu_topic) {
            auto imu = m.instantiate<sensor_msgs::Imu>();
            if (!imu) continue;
            // Every IMU sample goes to the backend: VINS owns buffering,
            // preintegration and the td offset. Unconditional now -- there is no
            // vision-only mode left to gate on.
            pipeline->addImu(imu->header.stamp.toSec(),
                cv::Vec3d(imu->linear_acceleration.x, imu->linear_acceleration.y, imu->linear_acceleration.z),
                cv::Vec3d(imu->angular_velocity.x, imu->angular_velocity.y, imu->angular_velocity.z));
            // GT attitude from the IMU orientation only when PPK/FRL isn't driving it.
            if (!use_ppk)
                cur_R = quatToRot(imu->orientation.x, imu->orientation.y,
                                  imu->orientation.z, imu->orientation.w);
            continue;
        }

        if (topic == bag_cfg.gps_topic) {
            if (use_ppk) continue;  // PPK overrides bag GPS entirely
            auto gps = m.instantiate<sensor_msgs::NavSatFix>();
            if (!gps) continue;
            if (std::isnan(gps->latitude) || std::isnan(gps->longitude)) continue;
            const double alt = std::isnan(gps->altitude) ? 0.0 : gps->altitude;

            if (!have_origin) {
                origin_lat = gps->latitude;
                origin_lon = gps->longitude;
                origin_alt = alt;
                have_origin = true;
            }
            cur_alt = alt;
            cur_position = enuOffset(origin_lat, origin_lon, origin_alt,
                                     gps->latitude, gps->longitude, alt);
            continue;
        }

        if (topic != bag_cfg.img_topic) continue;

        // PPK / FRL path: derive both position and attitude at the image
        // timestamp. lat/lon/alt feed the ENU position, roll/pitch/yaw (NED)
        // feed the GT rotation, so R and t come entirely from the FRL file.
        if (use_ppk) {
            double lat, lon, alt, roll, pitch, yaw;
            if (!interpolatePpk(ppk, (int64_t)ts_ns, lat, lon, alt, roll, pitch, yaw)) continue;
            if (!have_origin) {
                origin_lat = lat; origin_lon = lon; origin_alt = alt;
                have_origin = true;
            }
            cur_alt = alt;
            cur_position = enuOffset(origin_lat, origin_lon, origin_alt, lat, lon, alt);
            cur_R = eulerToRot(roll, pitch, yaw);
        }

        // Hold images until the first GPS / PPK packet pins the local origin.
        if (!have_origin) continue;

        auto img_msg = m.instantiate<sensor_msgs::Image>();
        cv::Mat frame = decodeImage(img_msg);
        if (frame.empty()) continue;
        if (camera) frame = camera->undistortImage(frame);

        // Image timestamp on the IMU clock: image_clock + td = imu_clock.
        // Raw image stamp: VINS applies its own configured td internally
        // (processMeasurements does curTime = t + td). Pre-shifting here would
        // apply it twice.
        const double frame_time = img_msg->header.stamp.toSec();

        GroundTruthData current_gt;
        current_gt.position = cur_position;
        float pitch, roll, yaw;
        PoseMath::extractEulerFromRotation(cur_R, pitch, roll, yaw);
        current_gt.orientation = cv::Vec3f(pitch, roll, yaw);
        // Altimeter (AGL) for homography metric scale; left NaN when disabled.
        if (bag_cfg.altimeter_scale)
            current_gt.altitude = (float)(cur_alt - baseline_agl);

        DeviceBuffer frame_buffer(frame);
        pipeline->processFrame(frame_buffer, current_gt, frame_time);

        if (pipeline->isTrackingActive()) {
            // Body-referenced VO pose in the display frame: extrinsic + the
            // handedness sign flip applied (conjugation by traj_flip keeps the
            // rotation valid), but not yet the yaw alignment.
            // getGlobalTransformVO() is already a BODY pose (VINS estimates the
            // IMU state), so cam0_T_body must NOT be applied -- the old pipeline
            // integrated a camera pose and needed it.
            cv::Mat M = traj_flip * pipeline->getGlobalTransformVO() * traj_flip;

            // Auto-aligner: record the VO and GT start once tracking begins, and
            // once GT has travelled aligner_init_distance, solve the yaw that
            // rotates the VO displacement onto the GT displacement. Held after.
            if (!align_ready) {
                cv::Point2d p_vo(M.at<double>(0, 3), M.at<double>(1, 3));
                cv::Point2d p_gt(current_gt.position[0], current_gt.position[1]);
                if (!align_start_set) {
                    vo_start = p_vo;
                    gt_start = p_gt;
                    align_start_set = true;
                }
                cv::Point2d gt_disp = p_gt - gt_start;
                if (cv::norm(gt_disp) >= bag_cfg.aligner_init_distance) {
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

            // Apply the yaw alignment outermost in the display frame.
            cv::Mat T_world = align_T_vo * M;
            float pred_x = T_world.at<double>(0, 3);
            float pred_y = T_world.at<double>(1, 3);
            float pred_z = T_world.at<double>(2, 3);

            cv::Mat R_pred = T_world(cv::Rect(0, 0, 3, 3));
            float qx, qy, qz, qw;
            PoseMath::rot2quat(R_pred, qx, qy, qz, qw);

            log_file << frame_id << "," << pred_x << "," << pred_y << "," << pred_z << ","
                     << qx << "," << qy << "," << qz << "," << qw << "\n";

            if (!no_gui) {
                cv::Vec3f est_xyz(pred_x, pred_y, pred_z);
                cv::Mat traj_img = trajectory_visualizer.update(est_xyz, current_gt.position);
                cv::imshow("Rosbag 2D Trajectory", traj_img);
            }
        }

        const double agl = cur_alt - baseline_agl;
        const PipelineMetrics& mtr = pipeline->getMetrics();
        printf("\r[%04d] AGL:%6.1fm | Det:%4.0fms | Mat:%4.0fms | Pos:%3.0fms | Tot:%4.0fms | FPS:%4.1f   ",
               frame_id, agl,
               mtr.time_detect_ms, mtr.time_match_ms, mtr.time_pose_ms,
               mtr.time_total_ms, mtr.fps);
        fflush(stdout);

        if (!no_gui) {
            cv::Mat vis = pipeline->getDebugFrame();
            if (!vis.empty()) {
                cv::imshow("Rosbag VO Feature Tracking", vis);
                if (cv::waitKey(1) == 27) break;
            }
        }

        frame_id++;
    }

    bag.close();
    printf("\n");
    log_file.close();
    std::cout << "[EVALUATOR] Complete. Trajectory saved to " << out_path << "\n";
    return 0;
}
