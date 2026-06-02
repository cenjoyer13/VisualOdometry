#include <iostream>
#include <fstream>
#include <thread>
#include <atomic>
#include <mutex>
#include <chrono>
#include <iomanip>
#include <string>

#include <Windows.h>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

#include "vehicles/multirotor/api/MultirotorRpcLibClient.hpp"

#include "odometry/utils/InputUtils.h"

using namespace msr::airlib;

struct UserInput {
    float vx = 0.0f, vy = 0.0f, vz = 0.0f;
    float yaw_rate = 0.0f;
};

struct RecorderContext {
    UserInput input;
    cv::Mat last_frame;
    std::mutex data_mutex;
    std::atomic<bool> has_new_frame{false};
    std::atomic<bool> is_running{true};
};

static void recorderWorker(RecorderContext* ctx, const std::string& out_path, int sample_hz) {
    try {
        MultirotorRpcLibClient client;
        client.confirmConnection();
        client.enableApiControl(true);
        client.armDisarm(true);
        client.takeoffAsync()->waitOnLastTask();

        std::ofstream out(out_path);
        if (!out.is_open()) {
            std::cerr << "[RECORDER] Failed to open output file: " << out_path << "\n";
            ctx->is_running = false;
            return;
        }

        out << std::fixed << std::setprecision(6);
        out << "# AirSim recorded path (NED frame; positions in meters)\n";
        out << "# Sampled at ~" << sample_hz << " Hz\n";
        out << "time_s,x,y,z,qw,qx,qy,qz\n";

        std::cout << "[RECORDER] Recording to " << out_path
                  << " at ~" << sample_hz << " Hz. Fly with WASD+Arrows.\n";

        auto start_time = std::chrono::steady_clock::now();
        auto last_sample = start_time - std::chrono::hours(1);
        auto last_cmd = start_time;
        const auto sample_period_ms = std::chrono::milliseconds(1000 / std::max(1, sample_hz));

        int sample_count = 0;

        while (ctx->is_running) {
            UserInput in;
            {
                std::lock_guard<std::mutex> lk(ctx->data_mutex);
                in = ctx->input;
            }

            auto now = std::chrono::steady_clock::now();

            // Send velocity command at ~20 Hz
            if (std::chrono::duration_cast<std::chrono::milliseconds>(now - last_cmd).count() > 50) {
                YawMode ym(true, in.yaw_rate);
                client.moveByVelocityBodyFrameAsync(in.vx, in.vy, in.vz, 0.15f,
                    DrivetrainType::MaxDegreeOfFreedom, ym);
                last_cmd = now;
            }

            // Stream image to UI for feedback
            std::vector<ImageCaptureBase::ImageRequest> req = {
                ImageCaptureBase::ImageRequest("0", ImageCaptureBase::ImageType::Scene, false, false)
            };
            auto resp = client.simGetImages(req);
            if (!resp.empty() && !resp[0].image_data_uint8.empty()) {
                cv::Mat raw(resp[0].height, resp[0].width, CV_8UC3,
                            (void*)resp[0].image_data_uint8.data());
                cv::Mat frame = raw.clone();
                std::lock_guard<std::mutex> lk(ctx->data_mutex);
                ctx->last_frame = frame;
                ctx->has_new_frame = true;
            }

            // Sample pose at fixed rate
            if (now - last_sample >= sample_period_ms) {
                MultirotorState st = client.getMultirotorState();
                const auto& p = st.kinematics_estimated.pose.position;
                const auto& q = st.kinematics_estimated.pose.orientation;
                double t = std::chrono::duration_cast<std::chrono::milliseconds>(now - start_time).count() / 1000.0;
                out << t << ","
                    << p.x() << "," << p.y() << "," << p.z() << ","
                    << q.w() << "," << q.x() << "," << q.y() << "," << q.z() << "\n";
                last_sample = now;
                sample_count++;
            }
        }

        out.flush();
        out.close();

        std::cout << "[RECORDER] Stopping. Captured " << sample_count << " samples. Landing...\n";
        client.landAsync()->waitOnLastTask();
        client.armDisarm(false);
        client.enableApiControl(false);
        std::cout << "[RECORDER] Trajectory saved: " << out_path << "\n";
    }
    catch (std::exception& e) {
        std::cerr << "[RECORDER ERROR] " << e.what() << "\n";
        ctx->is_running = false;
    }
}

int main(int argc, char** argv) {
    std::string out_path = (argc >= 2) ? argv[1] : "flight_path.csv";
    int sample_hz = 10;

    RecorderContext ctx;

    std::cout << "Launching path recorder thread...\n";
    std::thread worker(recorderWorker, &ctx, out_path, sample_hz);

    cv::Mat display;
    std::cout << "UI Ready. Controls:\n";
    std::cout << "  WASD+Arrows : Move\n";
    std::cout << "  Shift       : Turbo Speed\n";
    std::cout << "  ESC         : Stop recording, land, save\n";

    while (ctx.is_running) {
        UserInput in;
        float speed = isKeyPressed(VK_SHIFT) ? 10.0f : 3.0f;
        if (isKeyPressed('W')) in.vx = speed;
        if (isKeyPressed('S')) in.vx = -speed;
        if (isKeyPressed('D')) in.vy = speed;
        if (isKeyPressed('A')) in.vy = -speed;
        if (isKeyPressed(VK_UP)) in.vz = -speed;
        if (isKeyPressed(VK_DOWN)) in.vz = speed;
        if (isKeyPressed(VK_RIGHT)) in.yaw_rate = 60.0f;
        if (isKeyPressed(VK_LEFT)) in.yaw_rate = -60.0f;

        {
            std::lock_guard<std::mutex> lk(ctx.data_mutex);
            ctx.input = in;
            if (ctx.has_new_frame) {
                ctx.last_frame.copyTo(display);
                ctx.has_new_frame = false;
            }
        }

        if (!display.empty()) {
            cv::imshow("Recorder - Drone View", display);
        }
        if (cv::waitKey(10) == 27) {
            ctx.is_running = false;
        }
    }

    if (worker.joinable()) {
        worker.join();
    }
    return 0;
}
