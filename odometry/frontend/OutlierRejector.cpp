#include "OutlierRejector.h"

#include <algorithm>
#include <iostream>
#include <opencv2/calib3d.hpp>

namespace {
enum Model { kNone = 0, kEssential, kFundamental, kHomography };

int parseMethod(const std::string& s) {
    if (s == "essential")   return kEssential;
    if (s == "fundamental") return kFundamental;
    if (s == "homography")  return kHomography;
    return kNone;
}

// USAC variants are just method flags on the same OpenCV calls, so switching
// estimator costs nothing at the call site.
int parseUsac(const std::string& s) {
    if (s == "accurate") return cv::USAC_ACCURATE;
    if (s == "ransac")   return cv::RANSAC;
    return cv::USAC_MAGSAC;
}
}  // namespace

OutlierRejector::OutlierRejector(const OutlierRejectionParams& p)
    : p_(p), method_(parseMethod(p.method)), usac_flag_(parseUsac(p.usac)) {
    if (method_ != kNone) {
        std::cout << "[OutlierRejector] " << p_.method << " / " << p_.usac
                  << "  threshold " << p_.threshold_px << " px @ f="
                  << kVirtualFocal << "  conf " << p_.confidence
                  << "  min_points " << p_.min_points << "\n";
    }
}

OutlierRejector::Result OutlierRejector::run(const std::vector<cv::Point2d>& prev,
                                             const std::vector<cv::Point2d>& curr,
                                             std::vector<unsigned char>& mask) const {
    Result r;
    const size_t n = std::min(prev.size(), curr.size());
    mask.assign(n, 1);          // default: keep everything

    if (method_ == kNone)                    { r.skip_reason = "disabled";   return r; }
    if (static_cast<int>(n) < p_.min_points) { r.skip_reason = "too_few";    return r; }

    // Quoted in virtual pixels, applied on the normalised plane.
    const double thresh = p_.threshold_px / kVirtualFocal;

    std::vector<unsigned char> m;
    try {
        if (method_ == kEssential) {
            // Identity camera matrix: the points are already normalised, so the
            // threshold is a normalised-plane distance too.
            const cv::Mat K = cv::Mat::eye(3, 3, CV_64F);
            cv::Mat E = cv::findEssentialMat(prev, curr, K, usac_flag_,
                                             p_.confidence, thresh, m);
            if (E.empty()) { r.skip_reason = "no_model"; mask.assign(n, 1); return r; }
        } else if (method_ == kFundamental) {
            cv::Mat F = cv::findFundamentalMat(prev, curr, usac_flag_, thresh,
                                               p_.confidence, m);
            if (F.empty()) { r.skip_reason = "no_model"; mask.assign(n, 1); return r; }
        } else {
            cv::Mat H = cv::findHomography(prev, curr, usac_flag_, thresh, m,
                                           2000, p_.confidence);
            if (H.empty()) { r.skip_reason = "no_model"; mask.assign(n, 1); return r; }
        }
    } catch (const cv::Exception& e) {
        // A degenerate configuration must not cost the frame its features.
        r.skip_reason = "exception";
        mask.assign(n, 1);
        return r;
    }

    if (m.size() != n) { r.skip_reason = "bad_mask"; mask.assign(n, 1); return r; }

    int inliers = 0;
    for (size_t i = 0; i < n; ++i) inliers += (m[i] != 0);

    // A model that rejects almost everything is far more likely to be a bad fit
    // -- planar degeneracy, or near-pure rotation -- than a frame where almost
    // every track went bad. Discarding the frame's features on that basis would
    // be worse than doing nothing, so treat it as a skip.
    if (inliers < p_.min_points) {
        r.skip_reason = "degenerate";
        mask.assign(n, 1);
        return r;
    }

    mask = m;
    r.ran = true;
    r.n_in = inliers;
    r.n_out = static_cast<int>(n) - inliers;
    return r;
}
