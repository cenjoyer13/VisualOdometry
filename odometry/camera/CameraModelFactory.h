#pragma once
#include <memory>
#include "ICameraModel.h"
#include "CameraModelConfig.h"

// Builds the camera model named by CameraModelConfig::model_type. Returns
// nullptr for an empty/unknown model_type; the caller then keeps whatever
// intrinsics it already has and skips undistortion.
class CameraModelFactory {
public:
    static std::unique_ptr<ICameraModel> create(const CameraModelConfig& cfg);
};
