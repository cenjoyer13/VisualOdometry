#include <iostream>
#include <string>
#include <thread>

#include <Windows.h>
#include <opencv2/highgui.hpp>
#include <opencv2/core/utils/logger.hpp>

#include "core/DroneWorker.h"
#include "odometry/OdometryTypes.h"
#include "odometry/utils/ConfigLoader.h"
#include "odometry/utils/CudaPreload.h"

static bool isKeyPressed(int key) {
    return (GetAsyncKeyState(key) & 0x8000) != 0;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0]
                  << " <odometry_config.yaml> [--log <path>] [--debug]\n";
        return -1;
    }

    setvbuf(stdout, NULL, _IONBF, 0);
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_SILENT);

    std::string yaml_file;
    std::string log_path = "freeplay_log.csv";
    bool cli_debug = false;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--debug") {
            cli_debug = true;
        } else if (arg == "--log" && i + 1 < argc) {
            log_path = argv[++i];
        } else if (yaml_file.empty()) {
            yaml_file = arg;
        }
    }
    if (yaml_file.empty()) {
        std::cerr << "Missing YAML config path.\n";
        return -1;
    }

    // Defaults tuned for AirSim front camera (640x480, 90 deg FOV).
    SharedContext ctx;
    ctx.config.backend = ComputeBackend::CPU;
    ctx.config.intrinsics = {320.0f, 320.0f, 320.0f, 240.0f};
    ctx.log_path = log_path;

    ConfigLoader loader(yaml_file);
    if (!loader.isOpen()) {
        std::cerr << "Failed to open YAML: " << yaml_file
                  << " (did you add '%YAML:1.0' to the top of the file?)\n";
        return -1;
    }
    loader.loadOdometryConfig(ctx.config);
    if (cli_debug) ctx.config.verbose = true;

    if (ctx.config.backend == ComputeBackend::CUDA) {
        CudaPreload::init(ctx.config.verbose);
    }

    std::cout << "[MAIN] Detector=" << ctx.config.detector_type
              << " Matcher=" << ctx.config.matcher_type
              << " Backend=" << (int)ctx.config.backend
              << " Log=" << ctx.log_path << "\n";

    std::cout << "Launching Drone Thread...\n";
    std::thread workerThread(runDroneLogic, &ctx);

    cv::Mat display_frame;
    std::cout << "UI Ready. Controls:\n";
    std::cout << "  WASD+Arrows : Move\n";
    std::cout << "  Shift       : Turbo Speed\n";
    std::cout << "  [O]         : Toggle Visual Odometry\n";
    std::cout << "  ESC         : Stop, land, save log\n";

    bool odom_enabled = false;
    bool o_pressed_last = false;

    while (ctx.is_running) {
        DroneInput input;

        bool o_pressed = isKeyPressed('O');
        if (o_pressed && !o_pressed_last) {
            odom_enabled = !odom_enabled;
            std::cout << "Odometry Tracking: " << (odom_enabled ? "ON" : "OFF") << "\n";
        }
        o_pressed_last = o_pressed;

        input.enable_odometry = odom_enabled;

        float speed = isKeyPressed(VK_SHIFT) ? 10.0f : 3.0f;
        if (isKeyPressed('W')) input.vx = speed;
        if (isKeyPressed('S')) input.vx = -speed;
        if (isKeyPressed('D')) input.vy = speed;
        if (isKeyPressed('A')) input.vy = -speed;
        if (isKeyPressed(VK_UP))    input.vz = -speed;
        if (isKeyPressed(VK_DOWN))  input.vz = speed;
        if (isKeyPressed(VK_RIGHT)) input.yaw = 60.0f;
        if (isKeyPressed(VK_LEFT))  input.yaw = -60.0f;

        {
            std::lock_guard<std::mutex> lock(ctx.data_mutex);
            ctx.input = input;
            if (ctx.has_new_frame) {
                ctx.last_frame.copyTo(display_frame);
                ctx.has_new_frame = false;
            }
        }

        if (!display_frame.empty()) {
            cv::imshow("FreeplayDrone - Live View", display_frame);
        }

        if (cv::waitKey(10) == 27) {
            ctx.is_running = false;
        }
    }

    if (workerThread.joinable()) {
        std::cout << "Stopping drone...\n";
        workerThread.join();
    }
    return 0;
}
