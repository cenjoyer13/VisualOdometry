cmake_minimum_required(VERSION 3.18)
project(OdometryKittiTest LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# =========================================================================
# Eigen ABI: EIGEN_DONT_VECTORIZE must stay OFF.
#
# It used to be set globally to match a libgtsam.so built without
# -march=native: GTSAM's compiled Eigen types used 16-byte SSE alignment while
# ours would have picked 32-byte AVX, and the mismatch corrupted the heap.
#
# GTSAM is gone and the constraint inverted. libceres.so is built WITHOUT
# EIGEN_DONT_VECTORIZE, so defining it here would give our translation units a
# different sizeof/alignof for the same Eigen types that Ceres was compiled
# against -- the identical failure mode, pointing the other way. Symptom when
# it was still set: a hard crash inside VINS initialisation, immediately after
# "gyroscope bias initial calibration", i.e. the first heavily Eigen/Ceres
# code path the estimator reaches.
# =========================================================================
if(NOT CMAKE_CXX_FLAGS MATCHES "march=native")
    # Ceres is a system/conda build compiled for a baseline ISA; -march=native
    # on our side is fine as long as Eigen's vectorisation policy matches, which
    # it now does (both vectorise).
endif()

# =========================================================================
# Compile flag profile.
#
# GCC/Clang: -O3 -march=native -ffast-math comes from CMakePresets.json
# (preset: release-optimized on Linux, windows-msvc-release on Windows).
# MSVC: /O2 /fp:fast /arch:AVX2 — same intent, MSVC syntax.
# =========================================================================
if(MSVC)
    add_compile_options(/W3 /permissive- /Zc:__cplusplus)
    # AirSim + ONNX headers tend to emit -Wstrict-aliasing-style noise that's
    # not actionable from our side; keep warnings on but don't escalate.
endif()

# =========================================================================
# OpenCV (>= 4.8.0)
# =========================================================================
find_package(OpenCV 4.8.0 REQUIRED)
message(STATUS "Found OpenCV: ${OpenCV_VERSION} at ${OpenCV_DIR}")
message(STATUS "OpenCV CUDA modules intentionally disabled. ONNX Runtime owns GPU inference.")

# =========================================================================
# ONNX Runtime — optional (USE_ONNX, default ON).
#
# Drives the deep detectors/matchers (SuperPointDetector, LightGlueMatcher)
# and their CUDA execution provider. USE_ONNX=OFF produces a pure CPU build
# with NO ONNX / CUDA / cuDNN dependency: those two classes and the CudaPreload
# bootstrap are dropped from the build, the USE_ONNX compile define is absent,
# and the factories fall back to the OpenCV detectors/matchers (ORB/SIFT,
# Kinematic/FLANN). The deep configs (SuperPoint/LightGlue) are then unavailable.
#
# Linux/macOS default: /usr/local/onnxruntime (install convention used here).
# Windows: -DONNXRUNTIME_DIR=C:/path/to/onnxruntime-win-x64-gpu-1.18.1 must
# be passed; we fail loudly otherwise.
# =========================================================================
option(USE_ONNX "Build the ONNX Runtime detectors/matchers (SuperPoint, LightGlue)" ON)

if(USE_ONNX)
    if(WIN32)
        set(ONNXRUNTIME_DIR "" CACHE PATH "Path to ONNX Runtime install (release ZIP root)")
        if(NOT ONNXRUNTIME_DIR)
            message(FATAL_ERROR
                "ONNXRUNTIME_DIR not set. Download an ONNX Runtime GPU release "
                "(https://github.com/microsoft/onnxruntime/releases) and configure with "
                "-DONNXRUNTIME_DIR=C:/path/to/onnxruntime-win-x64-gpu-<version>")
        endif()
    else()
        set(ONNXRUNTIME_DIR "/usr/local/onnxruntime" CACHE PATH "Path to ONNX Runtime install")
    endif()

    include_directories(${ONNXRUNTIME_DIR}/include)
    find_library(ONNXRUNTIME_LIB onnxruntime PATHS ${ONNXRUNTIME_DIR}/lib NO_DEFAULT_PATH)
    if(NOT ONNXRUNTIME_LIB)
        message(FATAL_ERROR "Could not find ONNX Runtime library in ${ONNXRUNTIME_DIR}/lib")
    endif()
    add_compile_definitions(USE_ONNX)
    message(STATUS "ONNX Runtime: ${ONNXRUNTIME_LIB}")
else()
    set(ONNXRUNTIME_LIB "")   # empty -> harmless in the target_link_libraries lines
    message(STATUS "USE_ONNX=OFF — CPU-only build: SuperPoint / LightGlue / CudaPreload excluded, no ONNX or CUDA dependency.")
endif()

# =========================================================================
# Ceres — the VINS backend's solver.
#
# Replaces GTSAM, which went out with the LocalBundleAdjustment. That also
# retires the EIGEN_DONT_VECTORIZE workaround's original justification (a
# libgtsam.so built without -march=native); it is kept for now because the
# vendored VINS tree has not been revalidated with Eigen vectorisation on, and
# flipping both at once would make any resulting corruption ambiguous.
# =========================================================================
# Located by path rather than find_package(Ceres). The Ceres builds available
# here live in conda envs whose CeresConfig.cmake chains find_dependency through
# SuiteSparse -> METIS -> OpenMP, each expecting the conda toolchain's own
# layout; resolving that against the system compiler is a long fight for no
# benefit, since VINS needs nothing from Ceres but <ceres/ceres.h> and
# libceres.so. Same pattern already used for ONNX Runtime and the ROS libs.
set(CERES_DIR "" CACHE PATH "Root of a Ceres install (expects include/ceres/ceres.h, lib/libceres.so)")
if(NOT CERES_DIR)
    set(CERES_DIR "${ROS_ROOT}")   # the conda ROS envs here ship Ceres alongside ROS
endif()
find_path(CERES_INCLUDE_DIR ceres/ceres.h  PATHS ${CERES_DIR}/include NO_DEFAULT_PATH)
find_library(CERES_LIB      ceres          PATHS ${CERES_DIR}/lib     NO_DEFAULT_PATH)
# Ceres's public headers include glog unconditionally in these builds.
find_path(GLOG_INCLUDE_DIR  glog/logging.h PATHS ${CERES_DIR}/include NO_DEFAULT_PATH)
find_library(GLOG_LIB       glog           PATHS ${CERES_DIR}/lib     NO_DEFAULT_PATH)
if(NOT CERES_INCLUDE_DIR OR NOT CERES_LIB)
    message(FATAL_ERROR
        "Ceres not found under ${CERES_DIR}. Pass -DCERES_DIR=<prefix> containing "
        "include/ceres/ceres.h and lib/libceres.so.")
endif()
# glog >= 0.7 requires consumers to declare how they link it; without this its
# headers hard-#error out. Ceres's public headers include glog unconditionally,
# so every target that touches Ceres needs the define.
find_path(EIGEN3_INCLUDE_DIR signature_of_eigen3_matrix_library
          PATHS ${CERES_DIR}/include/eigen3 /usr/include/eigen3 /usr/local/include/eigen3)
if(NOT EIGEN3_INCLUDE_DIR)
    message(FATAL_ERROR "Eigen3 not found. Ceres headers include <Eigen/Core> unprefixed.")
endif()
message(STATUS "Ceres: ${CERES_LIB}")
message(STATUS "Eigen3: ${EIGEN3_INCLUDE_DIR}")

# =========================================================================
# Odometry source library (shared by every executable target)
# =========================================================================
file(GLOB_RECURSE SRC_ODOMETRY CONFIGURE_DEPENDS "odometry/*.cpp")
if(NOT USE_ONNX)
    # Drop the ONNX detectors/matchers and the cuDNN bootstrap from the CPU build.
    list(FILTER SRC_ODOMETRY EXCLUDE REGEX "(SuperPointDetector|LightGlueMatcher|CudaPreload)\\.cpp$")
endif()

# =========================================================================
# Vendored VINS-Fusion backend (vins/).
#
# Copied from sendtooscar/VINS-Fusion-gpu with two deliberate deviations, both
# documented at their sites: PCL and the lidar-depth path are stripped, and
# utility/visualization.h is replaced by a headless stub. Everything else is
# upstream verbatim so the tree stays diffable against VINS-Fusion.
#
# It is still ROS-coupled (Estimator holds a ros::NodeHandle), which is why only
# the rosbag target can link it today. Stripping that is the prerequisite for
# running the backend outside a bag.
# =========================================================================
file(GLOB_RECURSE SRC_VINS CONFIGURE_DEPENDS
    "vins/vins_estimator/estimator/*.cpp"
    "vins/vins_estimator/factor/*.cpp"
    "vins/vins_estimator/initial/*.cpp"
    "vins/vins_estimator/featureTracker/*.cpp"
    "vins/vins_estimator/utility/*.cpp"
    "vins/camera_models/src/*/*.cc"
)
set(VINS_INCLUDE_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}/vins
    ${CMAKE_CURRENT_SOURCE_DIR}/vins/camera_models/include
)

# Split the odometry sources: only the VIO pipeline needs the VINS backend
# (and therefore Ceres + ROS). Frontend-only tools must not drag that in.
set(SRC_ODOMETRY_VIO ${SRC_ODOMETRY})
list(FILTER SRC_ODOMETRY_VIO INCLUDE REGEX "(OdometryPipeline|VinsBackend)\\.cpp$")
list(FILTER SRC_ODOMETRY     EXCLUDE REGEX "(OdometryPipeline|VinsBackend)\\.cpp$")

# =========================================================================
# KittiEvaluator / OdomLogEvaluator were dropped in the VINS switch.
#
# Both drove the pipeline through its old signature (build(config), a
# ground-truth-scaled processFrame) and neither has an IMU stream, so neither
# can feed a visual-inertial backend. main_odomlog.cpp is kept in the tree
# unbuilt: the odometry_log captures it replays carry a ~20 Hz MAVLink RAW_IMU,
# far below what preintegration needs, so reviving it is a separate exercise
# from this milestone. main_test.cpp (KITTI) was deleted outright.
# =========================================================================

# =========================================================================
# MatchDebugger — portable (Linux + Windows).
# Runs detector + matcher on frame pairs from an image folder and writes
# annotated PNGs (both frames side-by-side, matches drawn as lines, RANSAC
# inlier/outlier colouring). Frontend-only: no pipeline, no backend, no GT,
# which is why it links neither Ceres nor ROS.
# =========================================================================
add_executable(MatchDebugger
    main_matchdebug.cpp
    ${SRC_ODOMETRY}
)

target_link_libraries(MatchDebugger PRIVATE
    ${OpenCV_LIBS}
    ${ONNXRUNTIME_LIB}
    ${CMAKE_DL_LIBS}
)

if(UNIX)
    set_target_properties(MatchDebugger PROPERTIES
        INSTALL_RPATH "${ONNXRUNTIME_DIR}/lib:${CMAKE_CURRENT_SOURCE_DIR}/cuda12_libs"
        BUILD_WITH_INSTALL_RPATH TRUE
        LINK_FLAGS "-Wl,--disable-new-dtags")
endif()

# =========================================================================
# RosbagEvaluator — Linux-only, opt-in via -DROS_ROOT=<ROS Noetic install>.
#
# Replays a ROS1 bag through the odometry pipeline. Mirrors KittiEvaluator's
# CSV log schema. Needs rosbag + roscpp_serialization + sensor_msgs headers
# from a ROS Noetic install (system /opt/ros/noetic or a robostack conda
# env). Skipped silently when ROS_ROOT is unset.
# =========================================================================
set(ROS_ROOT "" CACHE PATH "Path to ROS Noetic install root (enables RosbagEvaluator)")
if(UNIX AND ROS_ROOT)
    find_library(ROSBAG_LIB           rosbag                 PATHS ${ROS_ROOT}/lib NO_DEFAULT_PATH)
    find_library(ROSBAG_STORAGE_LIB   rosbag_storage         PATHS ${ROS_ROOT}/lib NO_DEFAULT_PATH)
    find_library(ROSCPP_SER_LIB       roscpp_serialization   PATHS ${ROS_ROOT}/lib NO_DEFAULT_PATH)
    find_library(ROSTIME_LIB          rostime                PATHS ${ROS_ROOT}/lib NO_DEFAULT_PATH)
    find_library(ROSCONSOLE_LIB       rosconsole             PATHS ${ROS_ROOT}/lib NO_DEFAULT_PATH)
    find_library(CPP_COMMON_LIB       cpp_common             PATHS ${ROS_ROOT}/lib NO_DEFAULT_PATH)
    # roscpp + tf are needed by the vendored VINS estimator, not by bag reading:
    # its Estimator holds a ros::NodeHandle and estimator.h includes
    # tf/transform_listener.h. Both go away when the backend is de-ROSed.
    find_library(ROSCPP_LIB           roscpp                 PATHS ${ROS_ROOT}/lib NO_DEFAULT_PATH)
    find_library(TF_LIB               tf                     PATHS ${ROS_ROOT}/lib NO_DEFAULT_PATH)
    find_library(TF2_LIB              tf2                    PATHS ${ROS_ROOT}/lib NO_DEFAULT_PATH)
    find_library(TF2_ROS_LIB          tf2_ros                PATHS ${ROS_ROOT}/lib NO_DEFAULT_PATH)
    find_library(XMLRPCPP_LIB         xmlrpcpp               PATHS ${ROS_ROOT}/lib NO_DEFAULT_PATH)

    if(ROSBAG_LIB AND ROSBAG_STORAGE_LIB AND ROSCPP_SER_LIB AND ROSTIME_LIB)
        message(STATUS "ROS integration enabled — building RosbagEvaluator.")
        message(STATUS "  ROS_ROOT: ${ROS_ROOT}")

        add_executable(RosbagEvaluator
            main_rosbag.cpp
            ${SRC_ODOMETRY}
            ${SRC_ODOMETRY_VIO}
            ${SRC_VINS}
        )
        target_include_directories(RosbagEvaluator PRIVATE
            ${ROS_ROOT}/include
            ${VINS_INCLUDE_DIRS}
            ${CERES_INCLUDE_DIR}
            ${GLOG_INCLUDE_DIR}
            ${EIGEN3_INCLUDE_DIR}
        )
        # VINS includes <eigen3/Eigen/...>, Ceres includes <Eigen/...>; the
        # eigen3 dir above satisfies the latter and its parent the former.
        target_compile_definitions(RosbagEvaluator PRIVATE GLOG_USE_GLOG_EXPORT)
        target_link_libraries(RosbagEvaluator PRIVATE
            ${OpenCV_LIBS}
            ${ONNXRUNTIME_LIB}
            ${CERES_LIB}
            ${GLOG_LIB}
            ${ROSCPP_LIB}
            ${TF_LIB}
            ${TF2_LIB}
            ${TF2_ROS_LIB}
            ${XMLRPCPP_LIB}
            ${ROSBAG_LIB}
            ${ROSBAG_STORAGE_LIB}
            ${ROSCPP_SER_LIB}
            ${ROSTIME_LIB}
            ${ROSCONSOLE_LIB}
            ${CPP_COMMON_LIB}
            ${CMAKE_DL_LIBS}
        )
        # RPATH so the binary can find both ONNX RT / cuDNN AND the ROS libs
        # without LD_LIBRARY_PATH gymnastics.
        set_target_properties(RosbagEvaluator PROPERTIES
            INSTALL_RPATH "${ONNXRUNTIME_DIR}/lib:${CMAKE_CURRENT_SOURCE_DIR}/cuda12_libs:${ROS_ROOT}/lib:${CERES_DIR}/lib"
            BUILD_WITH_INSTALL_RPATH TRUE
            LINK_FLAGS "-Wl,--disable-new-dtags")
    else()
        message(WARNING "ROS_ROOT set but rosbag libraries not found under ${ROS_ROOT}/lib — skipping RosbagEvaluator.")
    endif()
else()
    if(UNIX)
        message(STATUS "RosbagEvaluator skipped: pass -DROS_ROOT=<ROS Noetic root> to enable.")
    endif()
endif()

# =========================================================================
# The AirSim targets (PathRecorder / PathPlayer / FreeplayDrone) and their
# core/ worker were removed. All three were Win32-only, all three drove the
# old ground-truth-scaled pipeline, and AirSim was already deprecated as a
# host in favour of recorded-bag replay.
# =========================================================================