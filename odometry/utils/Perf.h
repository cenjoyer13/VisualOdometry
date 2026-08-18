#pragma once
#include <chrono>
#include <cstdint>
#include <vector>

// Wall-clock breakdown of the replay loop.
//
// WHY THE WHOLE LOOP, NOT JUST THE PIPELINE
// -----------------------------------------
// The per-frame console readout only ever covered OdometryPipeline, so
// everything around it -- bag deserialisation, JPEG decode, undistortion, GT
// interpolation, CSV writing, GUI -- was invisible. That is how a loop can
// crawl while the pipeline cheerfully reports 50 FPS. This measures every
// stage and, critically, a Total that brackets the entire iteration, so
// sum(stages) vs Total exposes anything not yet attributed.
//
// Same structure as the loosely-coupled fork's LoopTimers (accumulate per
// stage, report periodically), extended two ways: samples are kept so the
// summary can report p50/p90/max rather than only a mean -- a stage that is
// usually 2 ms and occasionally 300 ms is a very different problem from one
// that is steadily 20 ms, and a mean hides that -- and each frame also emits a
// `perf` record into the structured run log so scripts/logreader.py can rank
// the breakdown without anyone reading console spam.

namespace perf {

// Fixed set rather than string keys: the hot path becomes an array index, and
// the stage list is small and known. Adding a stage means editing this enum
// and kStageNames together.
enum Stage {
    BagRead = 0,   // rosbag message deserialisation (MessageInstance::instantiate)
    Decode,        // compressed image -> cv::Mat
    Undistort,     // ICameraModel::undistortImage
    Gt,            // PPK interpolation + ENU conversion
    Frontend,      // IFrontend::process + promoteKeyframe
    Reject,        // two-view geometric outlier rejection
    Backend,       // VinsBackend::addFrame, i.e. VINS solve
    Imu,           // OdometryPipeline::addImu fan-out
    Log,           // trajectory CSV + RunLog writes
    Gui,           // imshow / trajectory visualiser
    Total,         // whole loop iteration; brackets all of the above
    kNumStages
};

extern const char* const kStageNames[kNumStages];

}  // namespace perf

class Perf {
public:
    static Perf& instance();

    void setEnabled(bool on) { enabled_ = on; }
    bool enabled() const { return enabled_; }
    // Console report cadence in frames. 0 disables the periodic console report
    // (the structured records and the final summary are unaffected).
    void setReportEvery(int n) { report_every_ = n; }

    void add(perf::Stage s, double ms);

    // Closes the current frame: emits the `perf` record and folds the frame's
    // stage times into the run totals. Call once per loop iteration.
    void endFrame();

    // Final console table + a `perf.summary` record.
    void finish();

private:
    Perf() = default;
    void reportConsole();

    bool enabled_ = false;
    int report_every_ = 300;
    long frames_ = 0;

    double cur_[perf::kNumStages] = {0};        // current frame
    double sum_[perf::kNumStages] = {0};        // whole run
    double since_report_[perf::kNumStages] = {0};
    std::vector<float> samples_[perf::kNumStages];  // float: 4 bytes x ~45k is nothing
};

// RAII stage timer. Cheap enough to leave in place: one steady_clock read at
// each end, and the whole thing short-circuits when Perf is disabled.
class PerfScope {
public:
    explicit PerfScope(perf::Stage s)
        : stage_(s), live_(Perf::instance().enabled()) {
        if (live_) t0_ = std::chrono::steady_clock::now();
    }
    ~PerfScope() {
        if (!live_) return;
        Perf::instance().add(stage_, std::chrono::duration<double, std::milli>(
                                         std::chrono::steady_clock::now() - t0_).count());
    }
    PerfScope(const PerfScope&) = delete;
    PerfScope& operator=(const PerfScope&) = delete;

private:
    perf::Stage stage_;
    bool live_;
    std::chrono::steady_clock::time_point t0_;
};

#define PERF_CAT_(a, b) a##b
#define PERF_CAT(a, b) PERF_CAT_(a, b)
#define PERF_SCOPE(stage) PerfScope PERF_CAT(_perf_scope_, __LINE__)(stage)
