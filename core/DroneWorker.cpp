#include "DroneWorker.h"
#include "../odometry/OdometryPipeline.h"
#include "../odometry/OdometryTypes.h"
#include "vehicles/multirotor/api/MultirotorRpcLibClient.hpp"

#include <chrono>
#include <thread>
#include <fstream> 
#include <cmath>    
#include <iostream>

using namespace msr::airlib;

// Helper to extract Euler angles from a 4x4 matrix for the CSV log
void extractEulerFromMatrix(const cv::Mat& T, float& pitch, float& roll, float& yaw) {
    if (T.empty()) { pitch = roll = yaw = 0; return; }
    
    double m00 = T.at<double>(0,0), m01 = T.at<double>(0,1), m02 = T.at<double>(0,2);
    double m10 = T.at<double>(1,0), m11 = T.at<double>(1,1), m12 = T.at<double>(1,2);
    double m20 = T.at<double>(2,0), m21 = T.at<double>(2,1), m22 = T.at<double>(2,2);

    float sy = std::sqrt(m00 * m00 + m10 * m10);
    bool singular = sy < 1e-6;

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

void runDroneLogic(SharedContext* ctx) {
    try {
        MultirotorRpcLibClient client;
        client.confirmConnection();
        client.enableApiControl(true);
        client.armDisarm(true);
        client.takeoffAsync()->waitOnLastTask();

        // Build the odometry pipeline from the YAML-loaded config provided
        // by main().
        auto pipeline = OdometryPipeline::build(ctx->config);

        std::ofstream log_file(ctx->log_path);
        log_file << "Time_s,FPS,Global_X_VO,Global_Y_VO,Global_Z_VO,Pitch_VO,Roll_VO,Yaw_VO,"
                 << "Global_X_VIO,Global_Y_VIO,Global_Z_VIO,Pitch_VIO,Roll_VIO,Yaw_VIO,"
                 << "True_X,True_Y,True_Z\n";

        std::cout << "[WORKER] System started. Recording data to " << ctx->log_path << "..." << std::endl;

        auto start_time = std::chrono::steady_clock::now();
        auto last_cmd_time = std::chrono::steady_clock::now();

        while (ctx->is_running) {
            auto loop_start = std::chrono::steady_clock::now();

            DroneInput local_input;
            {
                std::lock_guard<std::mutex> lock(ctx->data_mutex);
                local_input = ctx->input;
            }

            // Request Image
            std::vector<ImageCaptureBase::ImageRequest> request = {
                ImageCaptureBase::ImageRequest("0", ImageCaptureBase::ImageType::Scene, false, false)
            };
            std::vector<ImageCaptureBase::ImageResponse> response = client.simGetImages(request);

            if (!response.empty() && response[0].image_data_uint8.size() > 0) {
                cv::Mat raw_frame(response[0].height, response[0].width, CV_8UC3,
                                 (void*)response[0].image_data_uint8.data());
                cv::Mat frame = raw_frame.clone();

                if (!frame.empty()) {
                    // --- 1. EXTRACT GROUND TRUTH & IMU DATA ---
                    MultirotorState state = client.getMultirotorState();
                    auto q = state.kinematics_estimated.pose.orientation;
                    auto pos = state.kinematics_estimated.pose.position;

                    // Convert AirSim Quaternions to Euler Angles
                    float sinr_cosp = 2.0f * (q.w() * q.x() + q.y() * q.z());
                    float cosr_cosp = 1.0f - 2.0f * (q.x() * q.x() + q.y() * q.y());
                    float roll = std::atan2(sinr_cosp, cosr_cosp);

                    float sinp = 2.0f * (q.w() * q.y() - q.z() * q.x());
                    float pitch = (std::abs(sinp) >= 1.0f) ? std::copysign(CV_PI / 2.0f, sinp) : std::asin(sinp);

                    float siny_cosp = 2.0f * (q.w() * q.z() + q.x() * q.y());
                    float cosy_cosp = 1.0f - 2.0f * (q.y() * q.y() + q.z() * q.z());
                    float yaw = std::atan2(siny_cosp, cosy_cosp);

                    // Build GroundTruthData Struct for the pipeline
                    GroundTruthData current_gt;
                    current_gt.orientation = cv::Vec3f(pitch, roll, yaw);
                    current_gt.position = cv::Vec3f(pos.y(), pos.x(), -pos.z()); // Right/Forward/Up

                    // --- 2. RUN ODOMETRY PIPELINE ---
                    if (local_input.enable_odometry) {
                        DeviceBuffer frame_buffer(frame);
                        pipeline->processFrame(frame_buffer, current_gt);
                    }

                    // --- 3. SEND FRAME TO UI ---
                    {
                        std::lock_guard<std::mutex> lock(ctx->data_mutex);
                        ctx->last_frame = frame;
                        ctx->has_new_frame = true;
                    }
                }
            }

            // --- 4. EXECUTE MOVEMENT ---
            auto now = std::chrono::steady_clock::now();
            if (std::chrono::duration_cast<std::chrono::milliseconds>(now - last_cmd_time).count() > 50) {
                YawMode ym(true, local_input.yaw);
                client.moveByVelocityBodyFrameAsync(local_input.vx, local_input.vy, local_input.vz, 0.15f,
                    DrivetrainType::MaxDegreeOfFreedom, ym);
                last_cmd_time = now;
            }

            // --- 5. LOGGING ---
            double loop_duration = std::chrono::duration_cast<std::chrono::microseconds>(now - loop_start).count();
            double fps = (loop_duration > 0) ? (1000000.0 / loop_duration) : 0.0;
            double time_s = std::chrono::duration_cast<std::chrono::milliseconds>(now - start_time).count() / 1000.0;

            float gx_vo = 0, gy_vo = 0, gz_vo = 0, pitch_vo = 0, roll_vo = 0, yaw_vo = 0;
            float gx_vio = 0, gy_vio = 0, gz_vio = 0, pitch_vio = 0, roll_vio = 0, yaw_vio = 0;

            if (local_input.enable_odometry && pipeline->isTrackingActive()) {
                cv::Mat T_VO = pipeline->getGlobalTransformVO();
                cv::Mat T_VIO = pipeline->getGlobalTransformVIO();

                // Extract translations (Row 0,1,2 of Column 3)
                gx_vo = T_VO.at<double>(0, 3);
                gy_vo = T_VO.at<double>(1, 3);
                gz_vo = T_VO.at<double>(2, 3);
                extractEulerFromMatrix(T_VO, pitch_vo, roll_vo, yaw_vo);

                gx_vio = T_VIO.at<double>(0, 3);
                gy_vio = T_VIO.at<double>(1, 3);
                gz_vio = T_VIO.at<double>(2, 3);
                extractEulerFromMatrix(T_VIO, pitch_vio, roll_vio, yaw_vio);
            }

            // True Coordinates (Aligned with log formatting)
            msr::airlib::MultirotorState state = client.getMultirotorState();
            float true_x = state.kinematics_estimated.pose.position.y(); 
            float true_y = state.kinematics_estimated.pose.position.x(); 
            float true_z = -state.kinematics_estimated.pose.position.z();

            if (log_file.is_open()) {
                log_file << time_s << "," << fps << ","
                         << gx_vo << "," << gy_vo << "," << gz_vo << "," << pitch_vo << "," << roll_vo << "," << yaw_vo << ","
                         << gx_vio << "," << gy_vio << "," << gz_vio << "," << pitch_vio << "," << roll_vio << "," << yaw_vio << ","
                         << true_x << "," << true_y << "," << true_z << "\n";
            }
        }

        log_file.close();
        client.landAsync()->waitOnLastTask();
        client.armDisarm(false);
        client.enableApiControl(false);
        std::cout << "[WORKER] Log file saved. Exiting." << std::endl;
    }
    catch (std::exception& e) {
        std::cerr << "[WORKER ERROR] " << e.what() << std::endl;
    }
}
