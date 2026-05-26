#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <string>
#include <chrono>
#include <thread>
#include <cmath>

#include <Windows.h>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/utils/logger.hpp>

#include "vehicles/multirotor/api/MultirotorRpcLibClient.hpp"

#include "odometry/OdometryPipeline.h"
#include "odometry/OdometryTypes.h"
#include "odometry/utils/RealTime2DTrajectory.h"

using namespace msr::airlib;

struct Waypoint {
    double t;
    float x, y, z;       // NED
    float qw, qx, qy, qz;
};

static bool isKeyPressed(int key) {
    return (GetAsyncKeyState(key) & 0x8000) != 0;
}

static void extractEulerFromMatrix(const cv::Mat& T, float& pitch, float& roll, float& yaw) {
    if (T.empty()) { pitch = roll = yaw = 0; return; }
    double m00 = T.at<double>(0,0), m01 = T.at<double>(0,1);
    double m10 = T.at<double>(1,0), m11 = T.at<double>(1,1);
    double m20 = T.at<double>(2,0), m21 = T.at<double>(2,1), m22 = T.at<double>(2,2);
    float sy = std::sqrt(m00 * m00 + m10 * m10);
    bool singular = sy < 1e-6f;
    if (!singular) {
        pitch = std::asin(-m20);
        roll  = std::atan2(m21, m22);
        yaw   = std::atan2(m10, m00);
    } else {
        pitch = std::asin(-m20);
        roll  = 0;
        yaw   = std::atan2(-m01, m11);
    }
}

static void quatToEuler(float qw, float qx, float qy, float qz,
                        float& pitch, float& roll, float& yaw) {
    float sinr_cosp = 2.0f * (qw * qx + qy * qz);
    float cosr_cosp = 1.0f - 2.0f * (qx * qx + qy * qy);
    roll = std::atan2(sinr_cosp, cosr_cosp);
    float sinp = 2.0f * (qw * qy - qz * qx);
    pitch = (std::abs(sinp) >= 1.0f) ? std::copysign(static_cast<float>(CV_PI) / 2.0f, sinp)
                                     : std::asin(sinp);
    float siny_cosp = 2.0f * (qw * qz + qx * qy);
    float cosy_cosp = 1.0f - 2.0f * (qy * qy + qz * qz);
    yaw = std::atan2(siny_cosp, cosy_cosp);
}

static bool loadTrajectory(const std::string& path, std::vector<Waypoint>& out) {
    std::ifstream in(path);
    if (!in.is_open()) return false;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        if (!line.empty() && (line[0] == 't' || line[0] == 'T')) continue; // CSV header
        std::stringstream ss(line);
        std::string token;
        std::vector<double> vals;
        while (std::getline(ss, token, ',')) {
            try { vals.push_back(std::stod(token)); } catch (...) { vals.clear(); break; }
        }
        if (vals.size() < 4) continue;
        Waypoint wp;
        wp.t = vals[0];
        wp.x = static_cast<float>(vals[1]);
        wp.y = static_cast<float>(vals[2]);
        wp.z = static_cast<float>(vals[3]);
        if (vals.size() >= 8) {
            wp.qw = static_cast<float>(vals[4]);
            wp.qx = static_cast<float>(vals[5]);
            wp.qy = static_cast<float>(vals[6]);
            wp.qz = static_cast<float>(vals[7]);
        } else {
            wp.qw = 1.0f; wp.qx = wp.qy = wp.qz = 0.0f;
        }
        out.push_back(wp);
    }
    return !out.empty();
}

static bool loadOdometryConfig(const std::string& yaml_file, OdometryConfig& config,
                               float& playback_velocity) {
    cv::FileStorage fs(yaml_file, cv::FileStorage::READ);
    if (!fs.isOpened()) {
        std::cerr << "Failed to open YAML: " << yaml_file << "\n";
        std::cerr << "Did you add '%YAML:1.0' to the top of the file?\n";
        return false;
    }

    // Defaults tuned for AirSim front camera (640x480, 90 deg FOV).
    config.backend = ComputeBackend::CPU;
    config.intrinsics = {320.0f, 320.0f, 320.0f, 240.0f};
    config.pose_params["min_disparity"] = 2.0f;
    playback_velocity = 3.0f;

    config.detector_type = (std::string)fs["detector"]["type"];
    config.matcher_type = (std::string)fs["matcher"]["type"];

    cv::FileNode d_node = fs["detector"][config.detector_type];
    if (!d_node.empty()) {
        if (config.detector_type == "ORB") {
            config.detector_params["nfeatures"] = (float)(int)d_node["nfeatures"];
            config.detector_params["scale_factor"] = (float)d_node["scaleFactor"];
            config.detector_params["nlevels"] = (float)(int)d_node["nLevels"];
        } else if (config.detector_type == "SIFT") {
            config.detector_params["nfeatures"] = (float)(int)d_node["nfeatures"];
            config.detector_params["nOctaveLayers"] = (float)(int)d_node["nOctaveLayers"];
            config.detector_params["contrastThreshold"] = (float)d_node["contrastThreshold"];
            config.detector_params["edgeThreshold"] = (float)(int)d_node["edgeThreshold"];
            config.detector_params["sigma"] = (float)d_node["sigma"];
        }
    }

    if (!fs["matcher"].empty()) {
        config.matcher_params["ratio_thresh"] = (float)fs["matcher"]["distance_ratio"];
        if (config.matcher_type == "FLANN" && !fs["matcher"]["FLANN"].empty()) {
            config.matcher_params["kdTrees"] = (float)(int)fs["matcher"]["FLANN"]["kdTrees"];
            config.matcher_params["searchChecks"] = (float)(int)fs["matcher"]["FLANN"]["searchChecks"];
        }
    }

    if (!fs["bucketing"].empty()) {
        config.bucketing_params.enabled = (int)fs["bucketing"]["enabled"] != 0;
        config.bucketing_params.grid_cols = (int)fs["bucketing"]["grid_cols"];
        config.bucketing_params.grid_rows = (int)fs["bucketing"]["grid_rows"];
        config.bucketing_params.max_features_per_bucket = (int)fs["bucketing"]["max_features_per_bucket"];
    }

    if (!fs["local_bundle_adjustment"].empty()) {
        config.use_local_ba = (int)fs["local_bundle_adjustment"]["enabled"] != 0;
        config.lba_window_size = (int)fs["local_bundle_adjustment"]["window_size"];
        if (!fs["local_bundle_adjustment"]["opt_stride"].empty()) {
            config.lba_opt_stride = (int)fs["local_bundle_adjustment"]["opt_stride"];
        }
    }

    if (!fs["system"].empty()) {
        if (!fs["system"]["num_threads"].empty()) {
            config.num_threads = (int)fs["system"]["num_threads"];
        }
        std::string backend_str = "CPU";
        if (!fs["system"]["backend"].empty()) {
            backend_str = (std::string)fs["system"]["backend"];
        }
        if (backend_str == "CUDA")        config.backend = ComputeBackend::CUDA;
        else if (backend_str == "OPENCL") config.backend = ComputeBackend::OPENCL;
        else                              config.backend = ComputeBackend::CPU;
    }

    if (!fs["airsim"].empty()) {
        if (!fs["airsim"]["playback_velocity"].empty()) {
            playback_velocity = (float)fs["airsim"]["playback_velocity"];
        }
        cv::FileNode in_node = fs["airsim"]["intrinsics"];
        if (!in_node.empty()) {
            config.intrinsics.fx = (float)in_node["fx"];
            config.intrinsics.fy = (float)in_node["fy"];
            config.intrinsics.cx = (float)in_node["cx"];
            config.intrinsics.cy = (float)in_node["cy"];
        }
    }

    fs.release();
    return true;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <trajectory.csv> <odometry_config.yaml>\n";
        return -1;
    }

    setvbuf(stdout, NULL, _IONBF, 0);
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_SILENT);

    std::string traj_file = argv[1];
    std::string yaml_file = argv[2];

    // --- Load trajectory ---
    std::vector<Waypoint> waypoints;
    if (!loadTrajectory(traj_file, waypoints) || waypoints.size() < 2) {
        std::cerr << "Failed to load trajectory or too few waypoints: " << traj_file << "\n";
        return -1;
    }
    std::cout << "[PLAYER] Loaded " << waypoints.size() << " waypoints from " << traj_file << "\n";

    // --- Load odometry config ---
    OdometryConfig config;
    float playback_velocity = 3.0f;
    if (!loadOdometryConfig(yaml_file, config, playback_velocity)) {
        return -1;
    }
    std::cout << "[PLAYER] Detector=" << config.detector_type
              << " Matcher=" << config.matcher_type
              << " Playback velocity=" << playback_velocity << " m/s\n";

    auto pipeline = OdometryPipeline::build(config);

    // --- Open output log ---
    std::ofstream log("playback_log.csv");
    log << std::fixed << std::setprecision(6);
    log << "time_s,fps,"
        << "gt_x,gt_y,gt_z,gt_pitch,gt_roll,gt_yaw,"
        << "vo_x,vo_y,vo_z,vo_pitch,vo_roll,vo_yaw,"
        << "vio_x,vio_y,vio_z,vio_pitch,vio_roll,vio_yaw\n";

    RealTime2DTrajectory trajectory_visualizer(0.5f);

    try {
        MultirotorRpcLibClient client;
        std::cout << "[PLAYER] Connecting to AirSim...\n";
        client.confirmConnection();
        client.enableApiControl(true);
        client.armDisarm(true);
        client.takeoffAsync()->waitOnLastTask();

        // Move to first waypoint to establish a known starting pose
        const auto& wp0 = waypoints.front();
        std::cout << "[PLAYER] Moving to start ("
                  << wp0.x << ", " << wp0.y << ", " << wp0.z << ")...\n";
        client.moveToPositionAsync(wp0.x, wp0.y, wp0.z, 2.0f)->waitOnLastTask();
        std::this_thread::sleep_for(std::chrono::milliseconds(500));

        // Build path of remaining waypoints
        std::vector<Vector3r> path;
        path.reserve(waypoints.size());
        for (size_t i = 1; i < waypoints.size(); ++i) {
            path.emplace_back(waypoints[i].x, waypoints[i].y, waypoints[i].z);
        }

        std::cout << "[PLAYER] Starting path playback...\n";
        std::cout << "------------------------------------------------------\n";

        // Yaw aligns with direction of travel
        YawMode yaw_mode(false, 0.0f);
        client.moveOnPathAsync(path, playback_velocity,
                               /*timeout_sec=*/3600.0f,
                               DrivetrainType::ForwardOnly,
                               yaw_mode,
                               /*lookahead=*/-1.0f,
                               /*adaptive_lookahead=*/1.0f);

        auto start_time = std::chrono::steady_clock::now();
        const Vector3r& last_wp = path.back();
        bool aborted = false;
        int frame_id = 0;

        while (true) {
            auto loop_start = std::chrono::steady_clock::now();

            // --- Image capture ---
            std::vector<ImageCaptureBase::ImageRequest> req = {
                ImageCaptureBase::ImageRequest("0", ImageCaptureBase::ImageType::Scene, false, false)
            };
            auto resp = client.simGetImages(req);

            // --- Ground truth ---
            MultirotorState state = client.getMultirotorState();
            const auto& q = state.kinematics_estimated.pose.orientation;
            const auto& p = state.kinematics_estimated.pose.position;
            const auto& v = state.kinematics_estimated.linear_velocity;

            float gt_pitch, gt_roll, gt_yaw;
            quatToEuler(q.w(), q.x(), q.y(), q.z(), gt_pitch, gt_roll, gt_yaw);

            // Pipeline frame: X=East(right), Y=North(forward), Z=Up
            GroundTruthData current_gt;
            current_gt.orientation = cv::Vec3f(gt_pitch, gt_roll, gt_yaw);
            current_gt.position = cv::Vec3f(p.y(), p.x(), -p.z());

            // --- Odometry ---
            if (!resp.empty() && !resp[0].image_data_uint8.empty()) {
                cv::Mat raw(resp[0].height, resp[0].width, CV_8UC3,
                            (void*)resp[0].image_data_uint8.data());
                cv::Mat frame = raw.clone();
                DeviceBuffer buf(frame);
                pipeline->processFrame(buf, current_gt);
            }

            // --- VO / VIO state ---
            float vo_x = 0, vo_y = 0, vo_z = 0, vo_p = 0, vo_r = 0, vo_y_ang = 0;
            float vio_x = 0, vio_y = 0, vio_z = 0, vio_p = 0, vio_r = 0, vio_y_ang = 0;
            if (pipeline->isTrackingActive()) {
                cv::Mat T_VO  = pipeline->getGlobalTransformVO();
                cv::Mat T_VIO = pipeline->getGlobalTransformVIO();
                vo_x = T_VO.at<double>(0, 3);
                vo_y = T_VO.at<double>(1, 3);
                vo_z = T_VO.at<double>(2, 3);
                extractEulerFromMatrix(T_VO, vo_p, vo_r, vo_y_ang);
                vio_x = T_VIO.at<double>(0, 3);
                vio_y = T_VIO.at<double>(1, 3);
                vio_z = T_VIO.at<double>(2, 3);
                extractEulerFromMatrix(T_VIO, vio_p, vio_r, vio_y_ang);
            }

            // --- Visualization ---
            if (pipeline->isTrackingActive()) {
                cv::Vec3f est_xyz(vo_x, vo_y, vo_z);
                cv::Mat traj_img = trajectory_visualizer.update(est_xyz, current_gt.position);
                cv::imshow("Trajectory (Top-down) - VO vs GT", traj_img);
            }
            cv::Mat dbg = pipeline->getDebugFrame();
            if (!dbg.empty()) {
                cv::imshow("Playback VO Feature Tracking", dbg);
            }
            int key = cv::waitKey(1);
            if (key == 27) { aborted = true; break; }
            if (isKeyPressed(VK_ESCAPE)) { aborted = true; break; }

            // --- Log ---
            auto now = std::chrono::steady_clock::now();
            double t_s = std::chrono::duration_cast<std::chrono::milliseconds>(now - start_time).count() / 1000.0;
            double loop_us = std::chrono::duration_cast<std::chrono::microseconds>(now - loop_start).count();
            double fps = (loop_us > 0) ? (1000000.0 / loop_us) : 0.0;

            log << t_s << "," << fps << ","
                << current_gt.position[0] << "," << current_gt.position[1] << "," << current_gt.position[2] << ","
                << gt_pitch << "," << gt_roll << "," << gt_yaw << ","
                << vo_x << "," << vo_y << "," << vo_z << "," << vo_p << "," << vo_r << "," << vo_y_ang << ","
                << vio_x << "," << vio_y << "," << vio_z << "," << vio_p << "," << vio_r << "," << vio_y_ang
                << "\n";

            const PipelineMetrics& m = pipeline->getMetrics();
            printf("\r[%04d] Det:%4.0fms Mat:%4.0fms Pos:%3.0fms Tot:%4.0fms FPS:%4.1f   ",
                   frame_id, m.time_detect_ms, m.time_match_ms, m.time_pose_ms,
                   m.time_total_ms, m.fps);
            fflush(stdout);
            frame_id++;

            // --- Completion check: close to last waypoint AND nearly stopped ---
            double dx = p.x() - last_wp.x();
            double dy = p.y() - last_wp.y();
            double dz = p.z() - last_wp.z();
            double dist = std::sqrt(dx*dx + dy*dy + dz*dz);
            double speed = std::sqrt(v.x()*v.x() + v.y()*v.y() + v.z()*v.z());
            if (dist < 1.0 && speed < 0.3 && t_s > 2.0) {
                break;
            }
        }
        printf("\n");

        std::cout << "[PLAYER] " << (aborted ? "Aborted by user" : "Path complete") << ". Landing...\n";
        client.landAsync()->waitOnLastTask();
        client.armDisarm(false);
        client.enableApiControl(false);
    }
    catch (std::exception& e) {
        std::cerr << "\n[PLAYER ERROR] " << e.what() << "\n";
    }

    log.close();
    std::cout << "[PLAYER] Log saved to playback_log.csv\n";
    return 0;
}
