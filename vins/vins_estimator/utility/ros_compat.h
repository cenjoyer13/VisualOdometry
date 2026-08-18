#pragma once

// ROS-free replacements for the rosconsole / roscpp-assert macros the vendored
// VINS-Fusion sources use for logging and invariants.
//
// WHY THIS EXISTS
// ---------------
// After the ros::NodeHandle removal, the estimator used nothing functional from
// ROS: zero uses of sensor_msgs, nav_msgs, tf or ros::Publisher. What remained
// was ~100 call sites of ROS_INFO / ROS_WARN / ROS_DEBUG / ROS_ASSERT plus a
// pile of dead #includes, and those alone forced every consumer of this backend
// to compile and link against a ROS installation.
//
// Including this instead means the estimator links no ROS libraries at all. ROS
// then survives only in the bag reader (RosbagEvaluator), which is a testing
// target -- so a shipped build that reads its own capture format needs no ROS.
//
// Keeping the macro NAMES rather than rewriting the call sites is deliberate:
// the vendored tree stays diffable against upstream VINS-Fusion, so upstream
// fixes remain easy to apply. Every change to vins/ is a deletion or a shim,
// never a rewrite.
//
// SEMANTICS
// ---------
//   * ROS_DEBUG / ROS_DEBUG_STREAM are compiled out by default. Upstream they
//     sit below rosconsole's default INFO threshold, so they never printed
//     anyway; keeping them silent preserves the observed behaviour and avoids
//     making the 38 debug sites (several inside per-frame loops) suddenly
//     dominate stdout. Define VINS_VERBOSE_DEBUG to turn them on.
//   * ROS_ASSERT aborts like its rosconsole counterpart. It guards real
//     invariants in the factors (Jacobian sizes, parameter-block counts), so it
//     must NOT become a no-op -- silently continuing past one corrupts the
//     solve rather than stopping it.
//   * The _STREAM forms take an ostream expression, matching rosconsole.

#include <cstdio>
#include <cstdlib>
#include <sstream>
#include <string>

// Declared, not included: the vendored tree stays free of this project's
// headers. Defined in odometry/utils/RunLog.cpp, which prints to stderr AND
// emits a {"type":"vins",...} record into the structured run log, stamped with
// the frame being processed.
//
// This is why the de-ROS pass pays off twice. Every diagnostic VINS emits --
// including the initialisation failure reasons at estimator.cpp:498/548/576/661
// and the Ceres iteration count at :1071 -- already flows through these macros,
// so routing them here captures the backend's entire diagnostic surface with no
// further edits to upstream sources.
namespace vins_log {
void logf(const char* level, const char* fmt, ...);
void logs(const char* level, const std::string& text);
}  // namespace vins_log

#define VINS_LOG_PRINTF(level, ...) vins_log::logf(level, __VA_ARGS__)

#define VINS_LOG_STREAM(level, expr)                   \
    do {                                               \
        std::ostringstream _vins_ss;                   \
        _vins_ss << expr;                              \
        vins_log::logs(level, _vins_ss.str());         \
    } while (0)

#define ROS_INFO(...)  VINS_LOG_PRINTF("INFO", __VA_ARGS__)
#define ROS_WARN(...)  VINS_LOG_PRINTF("WARN", __VA_ARGS__)
#define ROS_ERROR(...) VINS_LOG_PRINTF("ERROR", __VA_ARGS__)

#define ROS_INFO_STREAM(expr)  VINS_LOG_STREAM("INFO", expr)
#define ROS_WARN_STREAM(expr)  VINS_LOG_STREAM("WARN", expr)
#define ROS_ERROR_STREAM(expr) VINS_LOG_STREAM("ERROR", expr)

#ifdef VINS_VERBOSE_DEBUG
#define ROS_DEBUG(...)        VINS_LOG_PRINTF("DEBUG", __VA_ARGS__)
#define ROS_DEBUG_STREAM(expr) VINS_LOG_STREAM("DEBUG", expr)
#else
// Consume the arguments in an unevaluated context so they still have to compile
// (and so unused-variable warnings stay suppressed) without emitting anything.
#define ROS_DEBUG(...)         do { if (false) { VINS_LOG_PRINTF("DEBUG", __VA_ARGS__); } } while (0)
#define ROS_DEBUG_STREAM(expr) do { if (false) { VINS_LOG_STREAM("DEBUG", expr); } } while (0)
#endif

#define ROS_BREAK()                                                        \
    do {                                                                   \
        std::fprintf(stderr, "[FATAL] ROS_BREAK at %s:%d\n", __FILE__, __LINE__); \
        std::abort();                                                      \
    } while (0)

#define ROS_ASSERT(cond)                                                   \
    do {                                                                   \
        if (!(cond)) {                                                     \
            std::fprintf(stderr, "[FATAL] assertion failed: %s (%s:%d)\n", \
                         #cond, __FILE__, __LINE__);                       \
            std::abort();                                                  \
        }                                                                  \
    } while (0)

#define ROS_ASSERT_MSG(cond, ...)                                          \
    do {                                                                   \
        if (!(cond)) {                                                     \
            std::fprintf(stderr, "[FATAL] assertion failed: %s (%s:%d): ", \
                         #cond, __FILE__, __LINE__);                       \
            std::fprintf(stderr, __VA_ARGS__);                             \
            std::fprintf(stderr, "\n");                                    \
            std::abort();                                                  \
        }                                                                  \
    } while (0)
