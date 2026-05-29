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
#include "odometry/utils/ConfigLoader.h"
#include "odometry/utils/CudaPreload.h"
#include "odometry/utils/RealTime2DTrajectory.h"

// Rotation matrix to quaternion. Branch on the largest diagonal element to
// keep the divisor well away from zero.
static void rot2quat(const cv::Mat& R, float& qx, float& qy, float& qz, float& qw) {
    double tr = R.at<double>(0,0) + R.at<double>(1,1) + R.at<double>(2,2);
    if (tr > 0) {
        double S = sqrt(tr+1.0) * 2;
        qw = 0.25 * S;
        qx = (R.at<double>(2,1) - R.at<double>(1,2)) / S;
        qy = (R.at<double>(0,2) - R.at<double>(2,0)) / S;
        qz = (R.at<double>(1,0) - R.at<double>(0,1)) / S;
    } else if ((R.at<double>(0,0) > R.at<double>(1,1)) && (R.at<double>(0,0) > R.at<double>(2,2))) {
        double S = sqrt(1.0 + R.at<double>(0,0) - R.at<double>(1,1) - R.at<double>(2,2)) * 2;
        qw = (R.at<double>(2,1) - R.at<double>(1,2)) / S;
        qx = 0.25 * S;
        qy = (R.at<double>(0,1) + R.at<double>(1,0)) / S;
        qz = (R.at<double>(0,2) + R.at<double>(2,0)) / S;
    } else if (R.at<double>(1,1) > R.at<double>(2,2)) {
        double S = sqrt(1.0 + R.at<double>(1,1) - R.at<double>(0,0) - R.at<double>(2,2)) * 2;
        qw = (R.at<double>(0,2) - R.at<double>(2,0)) / S;
        qx = (R.at<double>(0,1) + R.at<double>(1,0)) / S;
        qy = 0.25 * S;
        qz = (R.at<double>(1,2) + R.at<double>(2,1)) / S;
    } else {
        double S = sqrt(1.0 + R.at<double>(2,2) - R.at<double>(0,0) - R.at<double>(1,1)) * 2;
        qw = (R.at<double>(1,0) - R.at<double>(0,1)) / S;
        qx = (R.at<double>(0,2) + R.at<double>(2,0)) / S;
        qy = (R.at<double>(1,2) + R.at<double>(2,1)) / S;
        qz = 0.25 * S;
    }
}

// Pitch/roll/yaw from a rotation matrix (KITTI convention).
static void extractEulerFromRotation(const cv::Mat& R, float& pitch, float& roll, float& yaw) {
    float sy = std::sqrt(R.at<double>(0,0) * R.at<double>(0,0) + R.at<double>(1,0) * R.at<double>(1,0));
    bool singular = sy < 1e-6;
    if (!singular) {
        pitch = std::asin(-R.at<double>(2,0));
        roll  = std::atan2(R.at<double>(2,1), R.at<double>(2,2));
        yaw   = std::atan2(R.at<double>(1,0), R.at<double>(0,0));
    } else {
        pitch = std::asin(-R.at<double>(2,0));
        roll  = 0;
        yaw   = std::atan2(-R.at<double>(0,1), R.at<double>(1,1));
    }
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
    double lat, lon, alt;
};

// Parse RTKLib .pos lines. Format is space-separated:
//   YYYY/MM/DD HH:MM:SS.sss latitude longitude height Q ns sdn sde sdu ...
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

        int year, month, day, hour, minute;
        double sec, lat, lon, alt;
        int n = std::sscanf(line.c_str(),
            "%d/%d/%d %d:%d:%lf %lf %lf %lf",
            &year, &month, &day, &hour, &minute, &sec, &lat, &lon, &alt);
        if (n != 9) continue;

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
        out.push_back(s);
    }

    std::sort(out.begin(), out.end(),
        [](const PpkSample& a, const PpkSample& b){ return a.ts_ns < b.ts_ns; });
    std::cout << "[PPK] Loaded " << out.size() << " samples from " << path << "\n";
    return out;
}

// Linear interpolation; clamps to endpoints outside the sample window.
static bool interpolatePpk(const std::vector<PpkSample>& ppk, int64_t ts_ns,
                           double& lat, double& lon, double& alt) {
    if (ppk.empty()) return false;
    if (ts_ns <= ppk.front().ts_ns) {
        lat = ppk.front().lat; lon = ppk.front().lon; alt = ppk.front().alt;
        return true;
    }
    if (ts_ns >= ppk.back().ts_ns) {
        lat = ppk.back().lat; lon = ppk.back().lon; alt = ppk.back().alt;
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
    bool cli_debug = false;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--debug") cli_debug = true;
        else if (yaml_file.empty()) yaml_file = arg;
    }
    if (yaml_file.empty()) {
        std::cerr << "Usage: ./RosbagEvaluator <yaml_config_path> [--debug]\n";
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

    if (config.backend == ComputeBackend::CUDA) {
        CudaPreload::init(config.verbose);
    }

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

    auto pipeline = OdometryPipeline::build(config);

    std::ofstream log_file("rosbag_trajectory.csv");
    log_file << "Frame,Pred_X,Pred_Y,Pred_Z,Q_X,Q_Y,Q_Z,Q_W\n";

    std::cout << "[EVALUATOR] Starting rosbag replay: " << bag_cfg.bag_path << "\n";
    std::cout << "  img_topic=" << bag_cfg.img_topic
              << "  gps_topic=" << bag_cfg.gps_topic
              << "  imu_topic=" << bag_cfg.imu_topic << "\n";
    std::cout << "------------------------------------------------------\n";

    RealTime2DTrajectory trajectory_visualizer(0.5f);

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

        // PPK path: derive lat/lon/alt at the image timestamp.
        if (use_ppk) {
            double lat, lon, alt;
            if (!interpolatePpk(ppk, (int64_t)ts_ns, lat, lon, alt)) continue;
            if (!have_origin) {
                origin_lat = lat; origin_lon = lon; origin_alt = alt;
                have_origin = true;
            }
            cur_alt = alt;
            cur_position = enuOffset(origin_lat, origin_lon, origin_alt, lat, lon, alt);
        }

        // Hold images until the first GPS / PPK packet pins the local origin.
        if (!have_origin) continue;

        cv::Mat frame = decodeImage(m.instantiate<sensor_msgs::Image>());
        if (frame.empty()) continue;

        GroundTruthData current_gt;
        current_gt.position = cur_position;
        float pitch, roll, yaw;
        extractEulerFromRotation(cur_R, pitch, roll, yaw);
        current_gt.orientation = cv::Vec3f(pitch, roll, yaw);

        DeviceBuffer frame_buffer(frame);
        pipeline->processFrame(frame_buffer, current_gt);

        if (pipeline->isTrackingActive()) {
            cv::Mat T_VO = pipeline->getGlobalTransformVO();
            float pred_x = T_VO.at<double>(0, 3);
            float pred_y = T_VO.at<double>(1, 3);
            float pred_z = T_VO.at<double>(2, 3);

            cv::Mat R_pred = T_VO(cv::Rect(0, 0, 3, 3));
            float qx, qy, qz, qw;
            rot2quat(R_pred, qx, qy, qz, qw);

            log_file << frame_id << "," << pred_x << "," << pred_y << "," << pred_z << ","
                     << qx << "," << qy << "," << qz << "," << qw << "\n";

            cv::Vec3f est_xyz(pred_x, pred_y, pred_z);
            cv::Mat traj_img = trajectory_visualizer.update(est_xyz, current_gt.position);
            cv::imshow("Rosbag 2D Trajectory", traj_img);
        }

        const double agl = cur_alt - baseline_agl;
        const PipelineMetrics& mtr = pipeline->getMetrics();
        printf("\r[%04d] AGL:%6.1fm | Det:%4.0fms | Mat:%4.0fms | Pos:%3.0fms | Tot:%4.0fms | FPS:%4.1f   ",
               frame_id, agl,
               mtr.time_detect_ms, mtr.time_match_ms, mtr.time_pose_ms,
               mtr.time_total_ms, mtr.fps);
        fflush(stdout);

        cv::Mat vis = pipeline->getDebugFrame();
        if (!vis.empty()) {
            cv::imshow("Rosbag VO Feature Tracking", vis);
            if (cv::waitKey(1) == 27) break;
        }

        frame_id++;
    }

    bag.close();
    printf("\n");
    log_file.close();
    std::cout << "[EVALUATOR] Complete. Trajectory saved to rosbag_trajectory.csv\n";
    return 0;
}
