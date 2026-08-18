#pragma once
#include <string>
#include <cstdint>

// Structured run log: one JSON object per line (JSONL), written to
// <dir>/run.jsonl.
//
// WHY JSONL
// ---------
// Heterogeneous record types in one stream, no multi-line records to break
// grep, and still trivially parseable. A 4500-frame mun3 replay produces ~7 MB,
// which is nothing on disk -- the point is that scripts/logreader.py compresses
// it to a screenful rather than anyone reading it raw.
//
// Every record carries `type`, and (once a frame context is set) `frame` and
// `ts`, so records of different types join on the frame they belong to.
//
// The sink is a process-wide singleton because the vendored VINS tree reaches
// it through vins_log::logf/logs (see vins/vins_estimator/utility/ros_compat.h)
// and threading a handle through that code would mean editing the upstream
// sources, which this fork deliberately avoids.
class RunLog {
public:
    enum class Level { Off, Info, Debug, Trace };

    static RunLog& instance();

    // Opens <dir>/run.jsonl. Creates `dir` if needed. Level::Off disables all
    // writes cheaply (enabled() short-circuits before any formatting).
    // Safe to call once; a second call is ignored.
    void open(const std::string& dir, Level level);
    void close();

    bool enabled() const { return level_ != Level::Off && out_ != nullptr; }
    Level level() const { return level_; }
    static Level parseLevel(const std::string& s);

    // Frame context stamped onto every subsequent record. Set once per frame.
    void setFrame(int frame, double ts);
    int currentFrame() const { return frame_; }

    // Only every Nth frame emits per-frame records (run.meta / event / vins are
    // never strided). 1 = every frame.
    void setFrameStride(int n) { stride_ = n < 1 ? 1 : n; }
    bool frameSelected() const { return stride_ <= 1 || (frame_ % stride_) == 0; }

    // Appends one already-serialised record body (no braces, no frame fields).
    void writeRecord(const char* type, const std::string& body, bool stamp_frame);

    // FNV-1a of a file's contents, or "missing". Used by run.meta so a replay
    // can be tied to the exact config that produced it -- the wall-clock solver
    // budget incident showed that "same config" has to be provable, not assumed.
    static std::string fileHash(const std::string& path);

private:
    RunLog() = default;
    ~RunLog();
    RunLog(const RunLog&) = delete;
    RunLog& operator=(const RunLog&) = delete;

    void* out_ = nullptr;     // std::FILE*
    Level level_ = Level::Off;
    int frame_ = -1;
    double ts_ = 0.0;
    int stride_ = 1;
    long records_ = 0;
};

// Builds one record and writes it when it goes out of scope.
//
//   LogRec("feed")("dt", dt)("n_feat", n)("carry", carried / double(n));
//
// Cheap when logging is off: construction checks enabled() and every setter
// becomes a no-op, so call sites need no #ifdef or if-guard.
class LogRec {
public:
    // stamp_frame  : attach the current frame/ts (false for run-scoped records)
    // honor_stride : obey RunLog::setFrameStride. False for sparse, high-value
    //                records (VINS diagnostics, events) that must never be
    //                dropped by a stride meant to thin per-frame telemetry.
    explicit LogRec(const char* type, bool stamp_frame = true, bool honor_stride = true);
    ~LogRec();

    LogRec& operator()(const char* key, double v);
    LogRec& operator()(const char* key, int v);
    LogRec& operator()(const char* key, long v);
    LogRec& operator()(const char* key, bool v);
    LogRec& operator()(const char* key, const char* v);
    LogRec& operator()(const char* key, const std::string& v);
    // Fixed-length numeric array, e.g. a position or quaternion.
    LogRec& vec(const char* key, const double* v, int n);

private:
    void sep();

    const char* type_;
    bool stamp_frame_;
    bool live_;
    std::string body_;
};

// Reached from the vendored VINS tree. Declared here and in ros_compat.h so
// that tree needs no include of this project's headers -- only a symbol.
namespace vins_log {
// printf-style; prints to stderr and, when the run log is open, emits a
// {"type":"vins","level":...,"msg":...} record stamped with the current frame.
void logf(const char* level, const char* fmt, ...);
// Same, for the already-formatted ROS_*_STREAM forms.
void logs(const char* level, const std::string& text);
}  // namespace vins_log
