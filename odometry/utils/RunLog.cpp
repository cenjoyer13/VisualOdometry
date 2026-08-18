#include "RunLog.h"

#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <sys/stat.h>
#include <sys/types.h>

namespace {

// Minimal JSON string escaping: quotes, backslash and control characters.
// VINS messages are plain ASCII, but they do contain quotes.
std::string jsonEscape(const std::string& s) {
    std::string o;
    o.reserve(s.size() + 8);
    for (char c : s) {
        switch (c) {
            case '"':  o += "\\\""; break;
            case '\\': o += "\\\\"; break;
            case '\n': o += "\\n";  break;
            case '\r': o += "\\r";  break;
            case '\t': o += "\\t";  break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", c);
                    o += buf;
                } else {
                    o += c;
                }
        }
    }
    return o;
}

std::string num(double v) {
    // Ordinary scalars: plenty for positions, timings and ratios without
    // printing 17 digits everywhere.
    char buf[40];
    std::snprintf(buf, sizeof(buf), "%.9g", v);
    return buf;
}

// Timestamps need their own format. These are absolute epoch seconds (~1.65e9),
// where %.9g collapses to 1.65351282e+09 -- 0.01 s resolution, which destroys
// both the ability to join records by time and any feed-interval measurement.
// Fixed 6 decimals keeps microseconds.
std::string tnum(double v) {
    char buf[40];
    std::snprintf(buf, sizeof(buf), "%.6f", v);
    return buf;
}

}  // namespace

RunLog& RunLog::instance() {
    static RunLog inst;
    return inst;
}

RunLog::~RunLog() { close(); }

RunLog::Level RunLog::parseLevel(const std::string& s) {
    if (s == "off")   return Level::Off;
    if (s == "debug") return Level::Debug;
    if (s == "trace") return Level::Trace;
    return Level::Info;
}

void RunLog::open(const std::string& dir, Level level) {
    if (out_ != nullptr) return;          // already open
    level_ = level;
    if (level_ == Level::Off) return;

    ::mkdir(dir.c_str(), 0775);           // ignore EEXIST
    const std::string path = dir + "/run.jsonl";
    out_ = std::fopen(path.c_str(), "w");
    if (out_ == nullptr) {
        std::fprintf(stderr, "[RunLog] cannot open %s -- logging disabled\n", path.c_str());
        level_ = Level::Off;
        return;
    }
    std::fprintf(stderr, "[RunLog] %s\n", path.c_str());
}

void RunLog::close() {
    if (out_ != nullptr) {
        std::fflush(static_cast<std::FILE*>(out_));
        std::fclose(static_cast<std::FILE*>(out_));
        out_ = nullptr;
    }
    level_ = Level::Off;
}

void RunLog::setFrame(int frame, double ts) {
    frame_ = frame;
    ts_ = ts;
}

void RunLog::writeRecord(const char* type, const std::string& body, bool stamp_frame) {
    if (!enabled()) return;
    auto* f = static_cast<std::FILE*>(out_);

    std::fputc('{', f);
    std::fprintf(f, "\"type\":\"%s\"", type);
    if (stamp_frame && frame_ >= 0) {
        std::fprintf(f, ",\"frame\":%d,\"ts\":%s", frame_, tnum(ts_).c_str());
    }
    if (!body.empty()) {
        std::fputc(',', f);
        std::fwrite(body.data(), 1, body.size(), f);
    }
    std::fputs("}\n", f);

    // Flush periodically rather than per record: a killed run still yields a
    // usable log (which mattered when the harness SIGKILLed a replay mid-flight)
    // without paying an fsync per line.
    if ((++records_ % 200) == 0) std::fflush(f);
}

std::string RunLog::fileHash(const std::string& path) {
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (f == nullptr) return "missing";
    uint64_t h = 1469598103934665603ULL;          // FNV-1a offset basis
    unsigned char buf[8192];
    size_t n;
    while ((n = std::fread(buf, 1, sizeof(buf), f)) > 0) {
        for (size_t i = 0; i < n; ++i) {
            h ^= buf[i];
            h *= 1099511628211ULL;                // FNV prime
        }
    }
    std::fclose(f);
    char out[24];
    std::snprintf(out, sizeof(out), "%016llx", static_cast<unsigned long long>(h));
    return out;
}

// ---- LogRec ----------------------------------------------------------------

LogRec::LogRec(const char* type, bool stamp_frame, bool honor_stride)
    : type_(type), stamp_frame_(stamp_frame) {
    RunLog& L = RunLog::instance();
    live_ = L.enabled() && (!honor_stride || L.frameSelected());
}

LogRec::~LogRec() {
    if (live_) RunLog::instance().writeRecord(type_, body_, stamp_frame_);
}

void LogRec::sep() {
    if (!body_.empty()) body_ += ',';
}

LogRec& LogRec::operator()(const char* key, double v) {
    if (live_) { sep(); body_ += '"'; body_ += key; body_ += "\":"; body_ += num(v); }
    return *this;
}
LogRec& LogRec::operator()(const char* key, int v) {
    if (live_) { sep(); body_ += '"'; body_ += key; body_ += "\":"; body_ += std::to_string(v); }
    return *this;
}
LogRec& LogRec::operator()(const char* key, long v) {
    if (live_) { sep(); body_ += '"'; body_ += key; body_ += "\":"; body_ += std::to_string(v); }
    return *this;
}
LogRec& LogRec::operator()(const char* key, bool v) {
    if (live_) { sep(); body_ += '"'; body_ += key; body_ += "\":"; body_ += (v ? "true" : "false"); }
    return *this;
}
LogRec& LogRec::operator()(const char* key, const char* v) {
    return (*this)(key, std::string(v ? v : ""));
}
LogRec& LogRec::operator()(const char* key, const std::string& v) {
    if (live_) {
        sep();
        body_ += '"'; body_ += key; body_ += "\":\"";
        body_ += jsonEscape(v);
        body_ += '"';
    }
    return *this;
}
LogRec& LogRec::vec(const char* key, const double* v, int n) {
    if (live_) {
        sep();
        body_ += '"'; body_ += key; body_ += "\":[";
        for (int i = 0; i < n; ++i) {
            if (i) body_ += ',';
            body_ += num(v[i]);
        }
        body_ += ']';
    }
    return *this;
}

// ---- vins_log: the tee out of the vendored VINS tree ------------------------

namespace vins_log {

void logs(const char* level, const std::string& text) {
    std::fprintf(stderr, "[%s] %s\n", level, text.c_str());
    // Stamped with the frame (that is what makes "when did init start failing"
    // answerable) but exempt from the stride: VINS emits these sparsely and
    // they carry the initialisation failure reasons, so dropping them to thin
    // per-frame telemetry would defeat the purpose.
    LogRec r("vins", /*stamp_frame=*/true, /*honor_stride=*/false);
    r("level", level)("msg", text);
}

void logf(const char* level, const char* fmt, ...) {
    char buf[1024];
    va_list ap;
    va_start(ap, fmt);
    std::vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    logs(level, std::string(buf));
}

}  // namespace vins_log
