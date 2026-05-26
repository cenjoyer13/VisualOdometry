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
# ONNX Runtime — platform-aware lookup.
#
# Linux/macOS default: /usr/local/onnxruntime (install convention used here).
# Windows: -DONNXRUNTIME_DIR=C:/path/to/onnxruntime-win-x64-gpu-1.18.1 must
# be passed; we fail loudly otherwise.
# =========================================================================
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

if(WIN32)
    find_library(ONNXRUNTIME_LIB onnxruntime PATHS ${ONNXRUNTIME_DIR}/lib NO_DEFAULT_PATH)
else()
    find_library(ONNXRUNTIME_LIB onnxruntime PATHS ${ONNXRUNTIME_DIR}/lib NO_DEFAULT_PATH)
endif()

if(NOT ONNXRUNTIME_LIB)
    message(FATAL_ERROR "Could not find ONNX Runtime library in ${ONNXRUNTIME_DIR}/lib")
endif()
message(STATUS "ONNX Runtime: ${ONNXRUNTIME_LIB}")

# =========================================================================
# GTSAM
# =========================================================================
find_package(GTSAM REQUIRED)
include_directories(${GTSAM_INCLUDE_DIR})

# =========================================================================
# Odometry source library (shared by every executable target)
# =========================================================================
file(GLOB_RECURSE SRC_ODOMETRY CONFIGURE_DEPENDS "odometry/*.cpp")

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
    set_target_properties(KittiEvaluator PROPERTIES
        INSTALL_RPATH "${ONNXRUNTIME_DIR}/lib:${CMAKE_CURRENT_SOURCE_DIR}/cuda12_libs"
        BUILD_WITH_INSTALL_RPATH TRUE)
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
