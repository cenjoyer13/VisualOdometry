#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <string>
#include <cmath>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/utils/logger.hpp>

#include "odometry/OdometryPipeline.h"
#include "odometry/OdometryTypes.h"
#include "odometry/utils/RealTime2DTrajectory.h"

// Math helper: Rotation Matrix to Quaternion
void rot2quat(const cv::Mat& R, float& qx, float& qy, float& qz, float& qw) {
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

// ... [Keep extractEulerFromRotation and parseKittiPose unchanged] ...
void extractEulerFromRotation(const cv::Mat& R, float& pitch, float& roll, float& yaw) {
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

bool parseKittiPose(const std::string& line, GroundTruthData& gt_data) {
    std::istringstream iss(line);
    std::vector<double> values(12);
    for (int i = 0; i < 12; ++i) {
        if (!(iss >> values[i])) return false;
    }
    cv::Mat R = (cv::Mat_<double>(3, 3) << 
                 values[0], values[1], values[2],
                 values[4], values[5], values[6],
                 values[8], values[9], values[10]);

    float pitch, roll, yaw;
    extractEulerFromRotation(R, pitch, roll, yaw);
    gt_data.orientation = cv::Vec3f(pitch, roll, yaw);
    gt_data.position = cv::Vec3f(values[3], values[7], values[11]); 
    return true;
}

int main(int argc, char** argv) {
    // Only requires 1 argument now!
    if (argc < 2) {
        std::cerr << "Usage: ./KittiEvaluator <yaml_config_path>\n";
        return -1;
    }
    
    setvbuf(stdout, NULL, _IONBF, 0);
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_SILENT);
    
    std::string yaml_file = argv[1];

    // 1. Read YAML Configuration
    cv::FileStorage fs(yaml_file, cv::FileStorage::READ);
    if (!fs.isOpened()) {
        std::cerr << "Failed to open YAML file. Did you add '%YAML:1.0' to the top of the file?\n";
        return -1;
    }

    OdometryConfig config;
    config.backend = ComputeBackend::CPU;
    config.intrinsics = {718.856f, 718.856f, 607.192f, 185.215f}; // KITTI Seq 00
    
    // --- PARSE ALGORITHMS ---
    config.detector_type = (std::string)fs["detector"]["type"];
    config.matcher_type = (std::string)fs["matcher"]["type"];

    // --- PARSE DETECTOR PARAMS ---
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

    // --- PARSE MATCHER PARAMS ---
    config.matcher_params["ratio_thresh"] = (float)fs["matcher"]["distance_ratio"];
    if (config.matcher_type == "FLANN" && !fs["matcher"]["FLANN"].empty()) {
        config.matcher_params["kdTrees"] = (float)(int)fs["matcher"]["FLANN"]["kdTrees"];
        config.matcher_params["searchChecks"] = (float)(int)fs["matcher"]["FLANN"]["searchChecks"];
    }

    // --- PARSE BUCKETING PARAMS ---
    if (!fs["bucketing"].empty()) {
        config.bucketing_params.enabled = (int)fs["bucketing"]["enabled"] != 0;
        config.bucketing_params.grid_cols = (int)fs["bucketing"]["grid_cols"];
        config.bucketing_params.grid_rows = (int)fs["bucketing"]["grid_rows"];
        config.bucketing_params.max_features_per_bucket = (int)fs["bucketing"]["max_features_per_bucket"];
    }
    
    // Inside your config parser function
    if (!fs["local_bundle_adjustment"].empty()) {
        config.use_local_ba = (int)fs["local_bundle_adjustment"]["enabled"] != 0;
        config.lba_window_size = (int)fs["local_bundle_adjustment"]["window_size"];
    }
    
    // --- NEW: PARSE THREAD LIMIT ---
    if (!fs["system"]["num_threads"].empty()) {
        config.num_threads = (int)fs["system"]["num_threads"];
    }
    // --- Parse Core System Hardware Preferences ---
    std::string backend_str = "CPU"; // Safe default
    if (!fs["system"]["backend"].empty()) {
        backend_str = (std::string)fs["system"]["backend"];
    }

    // Map string to enum
    if (backend_str == "CUDA") {
        config.backend = ComputeBackend::CUDA;
    } else if (backend_str == "OPENCL") {
        config.backend = ComputeBackend::OPENCL;
    } else {
        config.backend = ComputeBackend::CPU;
    }
    
    config.pose_params["min_disparity"] = 2.0f;

    // --- EXTRACT PATHS FROM YAML ---
    std::string root_path = (std::string)fs["dataset"]["root_path"];
    std::string sequence = (std::string)fs["dataset"]["sequence"];
    
    std::string dataset_path = root_path + "/sequences/" + sequence + "/image_0";
    std::string poses_file = root_path + "/poses/" + sequence + ".txt"; // Derived Path!

    fs.release();

    // 2. Start Execution
    std::ifstream gt_stream(poses_file);
    if (!gt_stream.is_open()) {
        std::cerr << "Failed to open poses file: " << poses_file << "\n";
        return -1;
    }

    auto pipeline = OdometryPipeline::build(config);

    std::ofstream log_file("kitti_trajectory.csv");
    log_file << "Frame,Pred_X,Pred_Y,Pred_Z,Q_X,Q_Y,Q_Z,Q_W\n";

    int frame_id = 0;
    std::string gt_line;

    std::cout << "[EVALUATOR] Starting KITTI Sequence...\n";
    std::cout << "------------------------------------------------------\n";
    
    RealTime2DTrajectory trajectory_visualizer(0.5f);

    while (std::getline(gt_stream, gt_line)) {
        GroundTruthData current_gt;
        if (!parseKittiPose(gt_line, current_gt)) break;

        char img_name[256];
        snprintf(img_name, sizeof(img_name), "%s/%06d.png", dataset_path.c_str(), frame_id);
        
        cv::Mat frame = cv::imread(img_name, cv::IMREAD_GRAYSCALE);
        if (frame.empty()) break;

        DeviceBuffer frame_buffer(frame);
        
        pipeline->processFrame(frame_buffer, current_gt);

        if (pipeline->isTrackingActive()) {
            cv::Mat T_VO = pipeline->getGlobalTransformVO();
            float pred_x = T_VO.at<double>(0, 3);
            float pred_y = T_VO.at<double>(1, 3);
            float pred_z = T_VO.at<double>(2, 3);

            cv::Mat R = T_VO(cv::Rect(0, 0, 3, 3));
            float qx, qy, qz, qw;
            rot2quat(R, qx, qy, qz, qw);

            log_file << frame_id << "," << pred_x << "," << pred_y << "," << pred_z << ","
                     << qx << "," << qy << "," << qz << "," << qw << "\n";
                     
            // --- NEW: Send current positions to the visualizer ---
            cv::Vec3f est_xyz(pred_x, pred_y, pred_z);
            cv::Mat traj_img = trajectory_visualizer.update(est_xyz, current_gt.position);
            cv::imshow("KITTI 2D Trajectory", traj_img);
        }

        const PipelineMetrics& m = pipeline->getMetrics();
	// Compacted string (~55 characters) to completely avoid 80-column line-wrapping
        printf("\r[%04d] Det:%4.0fms | Mat:%4.0fms | Pos:%3.0fms | Tot:%4.0fms | FPS:%4.1f   ", 
               frame_id, 
               m.time_detect_ms, 
               m.time_match_ms, 
               m.time_pose_ms, 
               m.time_total_ms, 
               m.fps);
        fflush(stdout); // Manually command the OS to dump the buffer

        cv::Mat vis = pipeline->getDebugFrame();
        if (!vis.empty()) {
            cv::imshow("KITTI VO Feature Tracking", vis);
            if (cv::waitKey(1) == 27) break;
        }

        frame_id++;
    }
    
    printf("\n");
	
    log_file.close();
    std::cout << "\n[EVALUATOR] Complete. Trajectory saved to kitti_trajectory.csv\n";
    return 0;
}
