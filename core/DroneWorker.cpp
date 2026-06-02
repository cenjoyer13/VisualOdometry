#include "DroneWorker.h"
#include "../odometry/OdometryPipeline.h"
#include "../odometry/OdometryTypes.h"
#include "../odometry/utils/PoseMath.h"
#include "vehicles/multirotor/api/MultirotorRpcLibClient.hpp"

#include <chrono>
#include <thread>
#include <fstream>
#include <cmath>
#include <iostream>

using namespace msr::airlib;

void runDroneLogic(SharedContext* ctx) {
    try {
        MultirotorRpcLibClient client;
        client.confirmConnection();
        client.enableApiControl(true);
        client.armDisarm(true);
        client.takeoffAsync()->waitOnLastTask();

        // Pipeline is built from the YAML config that main() filled in.
        auto pipeline = OdometryPipeline::build(ctx->config);

        std::ofstream log_file(ctx->log_path);
        log_file << "time_s,fps,"
                 << "vo_x,vo_y,vo_z,"
                 << "gt_x,gt_y,gt_z,gt_pitch,gt_roll,gt_yaw\n";

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

            // Image capture.
            std::vector<ImageCaptureBase::ImageRequest> request = {
                ImageCaptureBase::ImageRequest("0", ImageCaptureBase::ImageType::Scene, false, false)
            };
            auto response = client.simGetImages(request);

            float gt_pitch = 0, gt_roll = 0, gt_yaw = 0;
            float gt_x = 0, gt_y = 0, gt_z = 0;

            if (!response.empty() && response[0].image_data_uint8.size() > 0) {
                cv::Mat raw_frame(response[0].height, response[0].width, CV_8UC3,
                                  (void*)response[0].image_data_uint8.data());
                cv::Mat frame = raw_frame.clone();

                if (!frame.empty()) {
                    MultirotorState state = client.getMultirotorState();
                    auto q = state.kinematics_estimated.pose.orientation;
                    auto pos = state.kinematics_estimated.pose.position;

                    PoseMath::quatToEuler(q.w(), q.x(), q.y(), q.z(), gt_pitch, gt_roll, gt_yaw);

                    GroundTruthData current_gt;
                    current_gt.orientation = cv::Vec3f(gt_pitch, gt_roll, gt_yaw);
                    current_gt.position = cv::Vec3f(pos.y(), pos.x(), -pos.z());  // Right / Forward / Up
                    gt_x = current_gt.position[0];
                    gt_y = current_gt.position[1];
                    gt_z = current_gt.position[2];

                    if (local_input.enable_odometry) {
                        DeviceBuffer frame_buffer(frame);
                        pipeline->processFrame(frame_buffer, current_gt);
                    }

                    {
                        std::lock_guard<std::mutex> lock(ctx->data_mutex);
                        ctx->last_frame = frame;
                        ctx->has_new_frame = true;
                    }
                }
            }

            // Motion command, rate-limited to ~20 Hz.
            auto now = std::chrono::steady_clock::now();
            if (std::chrono::duration_cast<std::chrono::milliseconds>(now - last_cmd_time).count() > 50) {
                YawMode ym(true, local_input.yaw);
                client.moveByVelocityBodyFrameAsync(local_input.vx, local_input.vy, local_input.vz, 0.15f,
                                                    DrivetrainType::MaxDegreeOfFreedom, ym);
                last_cmd_time = now;
            }

            // Logging.
            double loop_duration = std::chrono::duration_cast<std::chrono::microseconds>(now - loop_start).count();
            double fps = (loop_duration > 0) ? (1000000.0 / loop_duration) : 0.0;
            double time_s = std::chrono::duration_cast<std::chrono::milliseconds>(now - start_time).count() / 1000.0;

            float vo_x = 0, vo_y = 0, vo_z = 0;
            if (local_input.enable_odometry && pipeline->isTrackingActive()) {
                cv::Mat T_VO = pipeline->getGlobalTransformVO();
                vo_x = T_VO.at<double>(0, 3);
                vo_y = T_VO.at<double>(1, 3);
                vo_z = T_VO.at<double>(2, 3);
            }

            if (log_file.is_open()) {
                log_file << time_s << "," << fps << ","
                         << vo_x << "," << vo_y << "," << vo_z << ","
                         << gt_x << "," << gt_y << "," << gt_z << ","
                         << gt_pitch << "," << gt_roll << "," << gt_yaw << "\n";
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
