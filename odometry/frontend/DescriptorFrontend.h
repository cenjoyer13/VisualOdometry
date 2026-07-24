#pragma once
#include <memory>
#include <vector>
#include "IFrontend.h"
#include "../detectors/IFeatureDetector.h"
#include "../matchers/IFeatureMatcher.h"

// Detect-and-match frontend: the current pipeline behavior, extracted verbatim.
// Owns the detector + matcher and the keyframe anchor (image / keypoints /
// descriptors), plus a cache of the last processed frame so promoteKeyframe
// advances to it without re-detecting.
class DescriptorFrontend : public IFrontend {
public:
    DescriptorFrontend(const OdometryConfig& cfg,
                       std::unique_ptr<IFeatureDetector> detector,
                       std::unique_ptr<IFeatureMatcher> matcher);

    void initialize(DeviceBuffer& frame) override;
    FrontendResult process(DeviceBuffer& frame) override;
    void promoteKeyframe(DeviceBuffer& frame) override;

private:
    OdometryConfig config_;
    std::unique_ptr<IFeatureDetector> detector_;
    std::unique_ptr<IFeatureMatcher> matcher_;

    // Keyframe anchor.
    DeviceBuffer anchor_image_;
    std::vector<cv::KeyPoint> anchor_keypoints_;
    DeviceBuffer anchor_descriptors_;

    // Last frame processed (cached for promoteKeyframe).
    DeviceBuffer last_image_;
    std::vector<cv::KeyPoint> last_keypoints_;
    DeviceBuffer last_descriptors_;
};
