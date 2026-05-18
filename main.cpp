#include <iostream>
#include <thread>
#include <Windows.h>
#include <opencv2/highgui.hpp>

#include "core/DroneWorker.h"

bool isKeyPressed(int key) {
    return (GetAsyncKeyState(key) & 0x8000) != 0;
}

int main() {
    SharedContext ctx;

    std::cout << "Launching Drone Thread..." << std::endl;
    std::thread workerThread(runDroneLogic, &ctx);

    cv::Mat display_frame;
    std::cout << "UI Ready. Controls:" << std::endl;
    std::cout << "  WASD+Arrows : Move" << std::endl;
    std::cout << "  Shift       : Turbo Speed" << std::endl;
    std::cout << "  [O]         : Toggle Visual Odometry" << std::endl;
    std::cout << "  ESC         : Exit" << std::endl;

    bool odom_enabled = false;
    bool o_pressed_last = false;

    while (ctx.is_running) {
        DroneInput input;

        // Toggle Logic for O (Odometry)
        bool o_pressed = isKeyPressed('O');
        if (o_pressed && !o_pressed_last) {
            odom_enabled = !odom_enabled;
            std::cout << "Odometry Tracking: " << (odom_enabled ? "ON" : "OFF") << std::endl;
        }
        o_pressed_last = o_pressed;

        input.enable_odometry = odom_enabled;

        // Movement Logic
        float speed = isKeyPressed(VK_SHIFT) ? 10.0f : 3.0f;
        if (isKeyPressed('W')) input.vx = speed;
        if (isKeyPressed('S')) input.vx = -speed;
        if (isKeyPressed('D')) input.vy = speed;
        if (isKeyPressed('A')) input.vy = -speed;
        if (isKeyPressed(VK_UP)) input.vz = -speed;
        if (isKeyPressed(VK_DOWN)) input.vz = speed;
        if (isKeyPressed(VK_RIGHT)) input.yaw = 60.0f;
        if (isKeyPressed(VK_LEFT)) input.yaw = -60.0f;

        // Sync with worker thread
        {
            std::lock_guard<std::mutex> lock(ctx.data_mutex);
            ctx.input = input;

            if (ctx.has_new_frame) {
                ctx.last_frame.copyTo(display_frame);
                ctx.has_new_frame = false;
            }
        }

        if (!display_frame.empty()) {
            cv::imshow("Drone Control Center", display_frame);
        }

        if (cv::waitKey(10) == 27) { // ESC
            ctx.is_running = false;
        }
    }

    if (workerThread.joinable()) {
        std::cout << "Stopping drone..." << std::endl;
        workerThread.join();
    }

    return 0;
}
