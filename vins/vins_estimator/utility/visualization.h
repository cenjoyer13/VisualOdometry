#pragma once

// Headless replacement for VINS-Fusion's utility/visualization.h.
//
// Upstream, that header pulls in the whole ROS publishing stack (nav_msgs,
// visualization_msgs, CameraPoseVisualization, a dozen ros::Publisher globals)
// so the estimator can stream its state to RViz. This project consumes the
// estimator's state directly through Estimator's members instead, so none of
// that is wanted.
//
// Estimator.cpp references exactly one symbol from the original header --
// pubLatestOdometry(), called from inputIMU() on every IMU sample once the
// solver is in NON_LINEAR. Everything else it needs comes from its own headers.
// Keeping the include and no-op'ing the one call leaves estimator.cpp
// byte-identical to upstream, which matters: the vendored tree should stay
// diffable against VINS-Fusion so upstream fixes remain easy to apply.

#include <eigen3/Eigen/Dense>
#include <std_msgs/Header.h>

class Estimator;

// Everything estimator.cpp calls, no-op'd. processMeasurements() fires the
// whole publisher set once per solved frame; the state they would have
// serialised is read directly off the Estimator instead (see VinsBackend).
inline void pubLatestOdometry(const Eigen::Vector3d& /*P*/,
                              const Eigen::Quaterniond& /*Q*/,
                              const Eigen::Vector3d& /*V*/,
                              double /*t*/) {}

inline void printStatistics(const Estimator& /*estimator*/, double /*t*/) {}
inline void pubOdometry(const Estimator& /*e*/, const std_msgs::Header& /*h*/) {}
inline void pubKeyPoses(const Estimator& /*e*/, const std_msgs::Header& /*h*/) {}
inline void pubCameraPose(const Estimator& /*e*/, const std_msgs::Header& /*h*/) {}
inline void pubPointCloud(const Estimator& /*e*/, const std_msgs::Header& /*h*/) {}
inline void pubTF(const Estimator& /*e*/, const std_msgs::Header& /*h*/) {}
inline void pubKeyframe(const Estimator& /*e*/) {}
