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
#include "odometry/utils/ConfigLoader.h"
#include "odometry/utils/CudaPreload.h"
#include "odometry/utils/PoseMath.h"
#include "odometry/utils/RealTime2DTrajectory.h"

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
    PoseMath::extractEulerFromRotation(R, pitch, roll, yaw);
    gt_data.orientation = cv::Vec3f(pitch, roll, yaw);
    gt_data.position = cv::Vec3f(values[3], values[7], values[11]); 
    return true;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: ./KittiEvaluator <yaml_config_path> [--debug]\n";
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
        std::cerr << "Usage: ./KittiEvaluator <yaml_config_path> [--debug]\n";
        return -1;
    }

    // Defaults: KITTI Seq 00 intrinsics, CPU backend. YAML values overlay these.
    OdometryConfig config;
    config.backend = ComputeBackend::CPU;
    config.intrinsics = {718.856f, 718.856f, 607.192f, 185.215f};

    ConfigLoader loader(yaml_file);
    if (!loader.isOpen()) {
        std::cerr << "Failed to open YAML file. Did you add '%YAML:1.0' to the top of the file?\n";
        return -1;
    }
    loader.loadOdometryConfig(config);
    if (cli_debug) config.verbose = true;   // --debug overrides system.verbose.

    // Preload bundled CUDA / cuDNN libs so ONNX Runtime's CUDA EP loads
    // without a manual LD_LIBRARY_PATH export. Skipped on non-CUDA backends.
    if (config.backend == ComputeBackend::CUDA) {
        CudaPreload::init(config.verbose);
    }

    std::string root_path, sequence;
    if (!loader.loadKittiDataset(root_path, sequence)) {
        std::cerr << "YAML is missing the dataset block (root_path / sequence).\n";
        return -1;
    }

    std::string dataset_path = root_path + "/sequences/" + sequence + "/image_0";
    std::string poses_file = root_path + "/poses/" + sequence + ".txt";

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
            PoseMath::rot2quat(R, qx, qy, qz, qw);

            log_file << frame_id << "," << pred_x << "," << pred_y << "," << pred_z << ","
                     << qx << "," << qy << "," << qz << "," << qw << "\n";

            // Push the latest estimate and GT into the 2D plot.
            cv::Vec3f est_xyz(pred_x, pred_y, pred_z);
            cv::Mat traj_img = trajectory_visualizer.update(est_xyz, current_gt.position);
            cv::imshow("KITTI 2D Trajectory", traj_img);
        }

        const PipelineMetrics& m = pipeline->getMetrics();
        // Compact single-line status; widths chosen to stay under one terminal row.
        printf("\r[%04d] Det:%4.0fms | Mat:%4.0fms | Pos:%3.0fms | Tot:%4.0fms | FPS:%4.1f   ",
               frame_id,
               m.time_detect_ms,
               m.time_match_ms,
               m.time_pose_ms,
               m.time_total_ms,
               m.fps);
        fflush(stdout);

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
