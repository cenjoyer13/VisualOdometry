#include "Perf.h"
#include "RunLog.h"

#include <algorithm>
#include <cstdio>

namespace perf {
const char* const kStageNames[kNumStages] = {
    "bag_read", "decode", "undistort", "gt", "frontend",
    "backend", "imu", "log", "gui", "total"
};
}  // namespace perf

namespace {

float pctile(std::vector<float> v, double p) {
    if (v.empty()) return 0.0f;
    const size_t i = std::min(v.size() - 1,
                              static_cast<size_t>(p / 100.0 * (v.size() - 1) + 0.5));
    std::nth_element(v.begin(), v.begin() + i, v.end());
    return v[i];
}

}  // namespace

Perf& Perf::instance() {
    static Perf p;
    return p;
}

void Perf::add(perf::Stage s, double ms) {
    if (!enabled_) return;
    cur_[s] += ms;
}

void Perf::endFrame() {
    if (!enabled_) return;

    {
        LogRec r("perf");
        for (int i = 0; i < perf::kNumStages; ++i) {
            r(perf::kStageNames[i], cur_[i]);
        }
        // What Total covers that no stage claimed. A large value means the loop
        // is spending time somewhere uninstrumented, which is exactly the class
        // of problem this whole file exists to surface.
        double attributed = 0.0;
        for (int i = 0; i < perf::Total; ++i) attributed += cur_[i];
        r("unattributed", cur_[perf::Total] - attributed);
    }

    for (int i = 0; i < perf::kNumStages; ++i) {
        sum_[i] += cur_[i];
        since_report_[i] += cur_[i];
        samples_[i].push_back(static_cast<float>(cur_[i]));
        cur_[i] = 0.0;
    }
    ++frames_;

    if (report_every_ > 0 && (frames_ % report_every_) == 0) reportConsole();
}

void Perf::reportConsole() {
    const double n = static_cast<double>(report_every_);
    std::fprintf(stderr, "\n[Perf] last %d frames (ms/frame):", report_every_);
    for (int i = 0; i < perf::kNumStages; ++i) {
        if (i == perf::Total) continue;
        if (since_report_[i] <= 0.0) continue;
        std::fprintf(stderr, " %s=%.2f", perf::kStageNames[i], since_report_[i] / n);
    }
    std::fprintf(stderr, " | total=%.2f\n", since_report_[perf::Total] / n);
    for (int i = 0; i < perf::kNumStages; ++i) since_report_[i] = 0.0;
}

void Perf::finish() {
    if (!enabled_ || frames_ == 0) return;

    const double n = static_cast<double>(frames_);
    const double total_mean = sum_[perf::Total] / n;

    std::fprintf(stderr,
        "\n[Perf] %ld frames -- mean/p50/p90/max ms per frame, and share of loop\n",
        frames_);
    for (int i = 0; i < perf::kNumStages; ++i) {
        const double mean = sum_[i] / n;
        const double share = (total_mean > 0.0 && i != perf::Total)
                                 ? 100.0 * mean / total_mean : 0.0;
        std::fprintf(stderr, "   %-12s %7.2f %7.2f %7.2f %7.2f  %5.1f%%\n",
                     perf::kStageNames[i], mean,
                     pctile(samples_[i], 50), pctile(samples_[i], 90),
                     pctile(samples_[i], 100),
                     i == perf::Total ? 100.0 : share);
    }

    LogRec s("perf.summary", /*stamp_frame=*/false);
    s("frames", frames_);
    for (int i = 0; i < perf::kNumStages; ++i) {
        const std::string base = perf::kStageNames[i];
        s((base + "_mean").c_str(), sum_[i] / n);
        s((base + "_p50").c_str(), static_cast<double>(pctile(samples_[i], 50)));
        s((base + "_p90").c_str(), static_cast<double>(pctile(samples_[i], 90)));
        s((base + "_max").c_str(), static_cast<double>(pctile(samples_[i], 100)));
        s((base + "_sum").c_str(), sum_[i]);
    }
}
