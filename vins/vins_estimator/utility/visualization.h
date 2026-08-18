#pragma once
#include "ros_compat.h"

// Headless replacement for VINS-Fusion's utility/visualization.h.
//
// Upstream, that header pulls in the whole ROS publishing stack (nav_msgs,
// visualization_msgs, CameraPoseVisualization, a dozen ros::Publisher globals)
// so the estimator can stream its state to RViz. This project consumes the
// estimator's state directly through Estimator's members instead, so none of
// that is wanted.
//
// The publisher set estimator.cpp calls once per solved frame is no-op'd here.
// Upstream they take a std_msgs::Header; these take the timestamp as a plain
// double instead, so no ROS message type leaks into the backend. That is the
// only signature divergence from upstream in this file.

#include <eigen3/Eigen/Dense>
class Estimator;

// Everything estimator.cpp calls, no-op'd. processMeasurements() fires the
// whole publisher set once per solved frame; the state they would have
// serialised is read directly off the Estimator instead (see VinsBackend).
inline void pubLatestOdometry(const Eigen::Vector3d& /*P*/,
                              const Eigen::Quaterniond& /*Q*/,
                              const Eigen::Vector3d& /*V*/,
                              double /*t*/) {}

inline void printStatistics(const Estimator& /*e*/, double /*t*/) {}
inline void pubOdometry(const Estimator& /*e*/, double /*stamp*/) {}
inline void pubKeyPoses(const Estimator& /*e*/, double /*stamp*/) {}
inline void pubCameraPose(const Estimator& /*e*/, double /*stamp*/) {}
inline void pubPointCloud(const Estimator& /*e*/, double /*stamp*/) {}
inline void pubTF(const Estimator& /*e*/, double /*stamp*/) {}
inline void pubKeyframe(const Estimator& /*e*/) {}
