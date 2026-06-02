#pragma once
#include <string>

// Parsed camera-model parameters from the YAML "camera:" block. Shared by
// ConfigLoader (producer) and CameraModelFactory (consumer). Kept free of
// OpenCV types so it stays a plain value object.
//
// For KannalaBrandtCamera the intrinsics are mu/mv/u0/v0 and the four fisheye
// polynomial coefficients are k2/k3/k4/k5 (OpenCV fisheye D = [k2,k3,k4,k5]).
// For Pinhole the same mu/mv/u0/v0 are the pinhole fx/fy/cx/cy and the
// distortion coefficients are ignored.
struct CameraModelConfig {
    std::string model_type;        // "KannalaBrandtCamera" | "Pinhole" | ""
    double scale_factor = 1.0;     // output-resolution multiplier
    int    image_width  = 0;
    int    image_height = 0;
    double mu = 0.0, mv = 0.0;     // focal lengths (px)
    double u0 = 0.0, v0 = 0.0;     // principal point (px)
    double k2 = 0.0, k3 = 0.0, k4 = 0.0, k5 = 0.0;  // KB distortion
};
