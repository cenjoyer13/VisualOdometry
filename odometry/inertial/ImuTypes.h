#pragma once
#include <string>
#include <opencv2/core.hpp>

// IMU value types and configuration. Header-only; pulled into OdometryConfig.
// The whole IMU layer is opt-in: ImuMode::Off (the default) leaves the pipeline
// behaving exactly as the vision-only path.

enum class ImuMode {
    Off,    // no IMU (default) — vision-only, unchanged behavior
    Gyro,   // preintegrated gyro -> rotation prior in the LBA
    Vio     // full NavState + bias + CombinedImuFactor in the LBA
};

// A single IMU measurement. acc in m/s^2, gyr in rad/s, t in seconds.
struct ImuSample {
    double t = 0.0;
    cv::Vec3d acc{0.0, 0.0, 0.0};
    cv::Vec3d gyr{0.0, 0.0, 0.0};
};

// IMU noise / gravity / init parameters. Names mirror the VINS config keys so
// values transfer directly. Extrinsic is reused from the top-level body_T_cam0.
struct ImuParams {
    ImuMode mode = ImuMode::Off;
    std::string topic = "/imu/data";
    double acc_n = 0.1;        // accelerometer measurement noise density
    double gyr_n = 0.01;       // gyroscope measurement noise density
    double acc_w = 0.001;      // accelerometer bias random walk
    double gyr_w = 0.0001;     // gyroscope bias random walk
    double g_norm = 9.81;      // gravity magnitude (m/s^2)
    double td = 0.0;           // camera-IMU time offset: image_clock + td = imu_clock
    std::string init = "gt";   // initializer strategy: "gt" | "vins" (later)

    // gyro mode: sigma (rad) of the rotation-only BetweenFactor prior. Tighter
    // than the VO between_rot_sigma so the gyro sharpens relative rotation.
    double gyro_rot_sigma = 0.01;

    // IMU->camera rotation that expresses IMU-frame angular velocity in the
    // camera (VO) frame: omega_cam = R_cam_imu * omega_imu. Derived from the
    // top-level body_T_cam0 as rotation(body_T_cam0)^T. Identity means the IMU
    // is assumed aligned with the camera. Critical: the LBA/VO lives in the
    // camera frame, so the gyro must be rotated into it before use.
    cv::Matx33d R_cam_imu = cv::Matx33d::eye();

    bool enabled() const { return mode != ImuMode::Off; }
};
