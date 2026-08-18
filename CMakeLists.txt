cmake_minimum_required(VERSION 3.18)
project(OdometryKittiTest LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# =========================================================================
# Eigen ABI safety vs GTSAM.
#
# A pre-built libgtsam.so / gtsam.lib compiled without -march=native lays out
# fixed-size Eigen types (Pose3, Matrix4d, Cal3_S2, Values entries, ...) with
# the SSE default of 16-byte alignment. Our code with -march=native on an
# AVX-capable CPU otherwise bumps Eigen's struct alignment to 32 bytes; the
# two halves agree on type names but disagree on size/alignment, and the
# first runOptimization() call corrupts the heap.
#
# EIGEN_DONT_VECTORIZE pins Eigen's struct alignment to 16 bytes so its types
# match GTSAM's compiled layout. Eigen still uses AVX for arithmetic — it
# just emits unaligned loads/stores where necessary, which is the layout
# GTSAM itself was compiled against.
# =========================================================================
add_compile_definitions(EIGEN_DONT_VECTORIZE)

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
# GTSAM
# =========================================================================
find_package(GTSAM REQUIRED)
include_directories(${GTSAM_INCLUDE_DIR})

# GTSAM's own lib dir usually isn't on the system loader path (it's a
# hand-built install under claude-sandbox/, not a system package), so every
# RPATH below that links gtsam also needs this appended — otherwise the
# binary fails at launch with "libgtsam.so.4 => not found" unless
# LD_LIBRARY_PATH is set manually first.
get_filename_component(GTSAM_LIB_DIR "${GTSAM_INCLUDE_DIR}/../lib" ABSOLUTE)

# =========================================================================
# Odometry source library (shared by every executable target)
# =========================================================================
file(GLOB_RECURSE SRC_ODOMETRY CONFIGURE_DEPENDS "odometry/*.cpp")
if(NOT USE_ONNX)
    # Drop the ONNX detectors/matchers and the cuDNN bootstrap from the CPU build.
    list(FILTER SRC_ODOMETRY EXCLUDE REGEX "(SuperPointDetector|LightGlueMatcher|CudaPreload)\\.cpp$")
endif()

# =========================================================================
# KittiEvaluator — portable (Linux + Windows).
# Replays a KITTI sequence through the pipeline. No AirSim dependency.
# =========================================================================
add_executable(KittiEvaluator
    main_test.cpp
    ${SRC_ODOMETRY}
)

target_link_libraries(KittiEvaluator PRIVATE
    ${OpenCV_LIBS}
    ${ONNXRUNTIME_LIB}
    gtsam
    ${CMAKE_DL_LIBS}     # libdl on Unix, empty on Windows
)

if(UNIX)
    # RPATH: lets the binary find libonnxruntime.so and the bundled cuda12_libs
    # without LD_LIBRARY_PATH gymnastics. $ORIGIN/.. = repo root from build/.
    # --disable-new-dtags: emit old-style DT_RPATH instead of DT_RUNPATH.
    # DT_RUNPATH (the modern default) only resolves the *executable's own*
    # direct NEEDED entries; libgtsam.so.4 itself needs libmetis-gtsam.so /
    # libcephes-gtsam.so.1 from the same install dir, and as a grandchild
    # dependency it's only found via the executable's path list if that path
    # list is the (transitively-inherited) old-style RPATH.
    set_target_properties(KittiEvaluator PROPERTIES
        INSTALL_RPATH "${ONNXRUNTIME_DIR}/lib:${CMAKE_CURRENT_SOURCE_DIR}/cuda12_libs:${GTSAM_LIB_DIR}"
        BUILD_WITH_INSTALL_RPATH TRUE
        LINK_FLAGS "-Wl,--disable-new-dtags")
endif()

# =========================================================================
# OdomLogEvaluator — portable (Linux + Windows).
# Replays a bare folder of frame_NNNNNN.jpg images (no ground truth) through
# the pipeline. Used for sanity-checking new captures that have no pose/PPK
# source. No AirSim/ROS dependency.
# =========================================================================
add_executable(OdomLogEvaluator
    main_odomlog.cpp
    ${SRC_ODOMETRY}
)

target_link_libraries(OdomLogEvaluator PRIVATE
    ${OpenCV_LIBS}
    ${ONNXRUNTIME_LIB}
    gtsam
    ${CMAKE_DL_LIBS}
)

if(UNIX)
    set_target_properties(OdomLogEvaluator PROPERTIES
        INSTALL_RPATH "${ONNXRUNTIME_DIR}/lib:${CMAKE_CURRENT_SOURCE_DIR}/cuda12_libs:${GTSAM_LIB_DIR}"
        BUILD_WITH_INSTALL_RPATH TRUE
        LINK_FLAGS "-Wl,--disable-new-dtags")
endif()

# =========================================================================
# MatchDebugger — portable (Linux + Windows).
# Runs detector + matcher on frame pairs from an image folder and writes
# annotated PNGs (both frames side-by-side, matches drawn as lines, RANSAC
# inlier/outlier colouring). Frontend-only sanity check: no pipeline, no pose
# estimation, no ground truth.
# =========================================================================
add_executable(MatchDebugger
    main_matchdebug.cpp
    ${SRC_ODOMETRY}
)

target_link_libraries(MatchDebugger PRIVATE
    ${OpenCV_LIBS}
    ${ONNXRUNTIME_LIB}
    gtsam
    ${CMAKE_DL_LIBS}
)

if(UNIX)
    set_target_properties(MatchDebugger PROPERTIES
        INSTALL_RPATH "${ONNXRUNTIME_DIR}/lib:${CMAKE_CURRENT_SOURCE_DIR}/cuda12_libs:${GTSAM_LIB_DIR}"
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

    if(ROSBAG_LIB AND ROSBAG_STORAGE_LIB AND ROSCPP_SER_LIB AND ROSTIME_LIB)
        message(STATUS "ROS integration enabled — building RosbagEvaluator.")
        message(STATUS "  ROS_ROOT: ${ROS_ROOT}")

        add_executable(RosbagEvaluator
            main_rosbag.cpp
            ${SRC_ODOMETRY}
        )
        target_include_directories(RosbagEvaluator PRIVATE ${ROS_ROOT}/include)
        target_link_libraries(RosbagEvaluator PRIVATE
            ${OpenCV_LIBS}
            ${ONNXRUNTIME_LIB}
            gtsam
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
            INSTALL_RPATH "${ONNXRUNTIME_DIR}/lib:${CMAKE_CURRENT_SOURCE_DIR}/cuda12_libs:${ROS_ROOT}/lib:${GTSAM_LIB_DIR}"
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
# AirSim-dependent targets: PathRecorder + PathPlayer + FreeplayDrone.
#
# All three use Win32 APIs (GetAsyncKeyState, VK_* constants) and AirSim
# RPC. They are gated on (WIN32 AND AIRSIM_*). Configure with:
#   cmake -DAIRSIM_ROOT=C:/path/to/AirSim ...
# or set AIRSIM_INCLUDE_DIRS / AIRSIM_LIBS directly to override defaults.
# =========================================================================
set(AIRSIM_ROOT "" CACHE PATH "Path to AirSim repository root")
set(AIRSIM_INCLUDE_DIRS "" CACHE STRING "Explicit AirSim include dirs (overrides AIRSIM_ROOT defaults)")
set(AIRSIM_LIBS "" CACHE STRING "Explicit AirSim libraries (e.g. AirLib;rpc)")

if(AIRSIM_ROOT AND NOT AIRSIM_INCLUDE_DIRS)
    set(AIRSIM_INCLUDE_DIRS
        ${AIRSIM_ROOT}/AirLib/include
        ${AIRSIM_ROOT}/AirLib/deps/rpclib/include
        ${AIRSIM_ROOT}/AirLib/deps/eigen3
        ${AIRSIM_ROOT}/MavLinkCom/include
    )
endif()

if(AIRSIM_ROOT AND NOT AIRSIM_LIBS)
    find_library(AIRSIM_AIRLIB_LIB AirLib
        PATHS ${AIRSIM_ROOT}/AirLib/lib ${AIRSIM_ROOT}/build_release/output/lib
        NO_DEFAULT_PATH)
    find_library(AIRSIM_RPC_LIB rpc
        PATHS ${AIRSIM_ROOT}/external/rpclib/rpc-2.3.0/build
              ${AIRSIM_ROOT}/AirLib/deps/rpclib/lib
              ${AIRSIM_ROOT}/build_release/output/lib
        NO_DEFAULT_PATH)
    if(AIRSIM_AIRLIB_LIB AND AIRSIM_RPC_LIB)
        set(AIRSIM_LIBS ${AIRSIM_AIRLIB_LIB} ${AIRSIM_RPC_LIB})
    endif()
endif()

if(WIN32 AND AIRSIM_INCLUDE_DIRS AND AIRSIM_LIBS)
    message(STATUS "AirSim integration enabled — building PathRecorder, PathPlayer, FreeplayDrone.")
    message(STATUS "  Includes: ${AIRSIM_INCLUDE_DIRS}")
    message(STATUS "  Libs:     ${AIRSIM_LIBS}")

    # ----- PathRecorder: WASD flight in AirSim, logs GT trajectory CSV -----
    add_executable(PathRecorder main_recorder.cpp)
    target_include_directories(PathRecorder PRIVATE ${AIRSIM_INCLUDE_DIRS})
    target_link_libraries(PathRecorder PRIVATE
        ${OpenCV_LIBS}
        ${AIRSIM_LIBS}
    )

    # ----- PathPlayer: replays a CSV trajectory with the odometry pipeline -----
    add_executable(PathPlayer
        main_player.cpp
        ${SRC_ODOMETRY}
    )
    target_include_directories(PathPlayer PRIVATE ${AIRSIM_INCLUDE_DIRS})
    target_link_libraries(PathPlayer PRIVATE
        ${OpenCV_LIBS}
        ${ONNXRUNTIME_LIB}
        gtsam
        ${AIRSIM_LIBS}
        ${CMAKE_DL_LIBS}
    )

    # ----- FreeplayDrone: interactive WASD flight + live odometry + CSV log -----
    add_executable(FreeplayDrone
        main.cpp
        core/DroneWorker.cpp
        ${SRC_ODOMETRY}
    )
    target_include_directories(FreeplayDrone PRIVATE ${AIRSIM_INCLUDE_DIRS})
    target_link_libraries(FreeplayDrone PRIVATE
        ${OpenCV_LIBS}
        ${ONNXRUNTIME_LIB}
        gtsam
        ${AIRSIM_LIBS}
        ${CMAKE_DL_LIBS}
    )
else()
    if(NOT WIN32)
        message(STATUS "AirSim targets skipped: PathRecorder / PathPlayer / FreeplayDrone are Win32-only.")
    else()
        message(STATUS "AirSim targets skipped: pass -DAIRSIM_ROOT=<repo root> to enable.")
    endif()
endif()
