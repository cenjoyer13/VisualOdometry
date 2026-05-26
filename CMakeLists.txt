cmake_minimum_required(VERSION 3.18) 
project(OdometryKittiTest LANGUAGES CXX) 

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -fno-omit-frame-pointer")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fsanitize=address")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native")
add_compile_definitions(EIGEN_DONT_ALIGN_STATICALLY EIGEN_DONT_VECTORIZE)

# --- FIND OPENCV 4.8.0 ---
find_package(OpenCV 4.8.0 REQUIRED)
message(STATUS "Found OpenCV: ${OpenCV_VERSION} at ${OpenCV_DIR}")

# --- OPENCV CUDA DISABLED ---
# We intentionally do not define HAS_CUDA here. 
# This prevents OpenCV from looking for missing contrib headers.
# LightGlue will still run on the GPU via ONNX Runtime's independent CUDA engine!
message(STATUS "OpenCV CUDA modules disabled. Neural Network CUDA enabled via ONNX.")

# --- ONNX RUNTIME SETUP (STRICT ABSOLUTE PATH) ---
set(ONNXRUNTIME_DIR "/usr/local/onnxruntime")
include_directories(${ONNXRUNTIME_DIR}/include)

find_library(ONNXRUNTIME_LIB onnxruntime PATHS ${ONNXRUNTIME_DIR}/lib NO_DEFAULT_PATH)
if(NOT ONNXRUNTIME_LIB)
    message(FATAL_ERROR "Could not find libonnxruntime.so in ${ONNXRUNTIME_DIR}/lib!")
endif()

# --- FIND GTSAM ---
find_package(GTSAM REQUIRED)
include_directories(${GTSAM_INCLUDE_DIR})

# --- GATHER ODOMETRY SOURCE FILES ---
file(GLOB_RECURSE SRC_ODOMETRY "odometry/*.cpp")

# --- BUILD EXECUTABLE ---
add_executable(KittiEvaluator
    main_test.cpp
    ${SRC_ODOMETRY}
)

target_link_libraries(KittiEvaluator PRIVATE
    ${OpenCV_LIBS}
    ${ONNXRUNTIME_LIB}
    gtsam
)

# --- BURN RPATHS INTO BINARY ---
# We use $ORIGIN/.. to make the path relative to your build/ directory,
# ensuring it always finds the cuda12_libs folder in your project root!
set_target_properties(KittiEvaluator PROPERTIES INSTALL_RPATH "${ONNXRUNTIME_DIR}/lib:${CMAKE_CURRENT_SOURCE_DIR}/cuda12_libs")
set_target_properties(KittiEvaluator PROPERTIES BUILD_WITH_INSTALL_RPATH TRUE)

# =========================================================================
# AirSim-dependent targets: PathRecorder + PathPlayer
# Both require AirSim headers and libs. Configure with:
#   cmake -DAIRSIM_ROOT=/path/to/AirSim ...
# (or set AIRSIM_INCLUDE_DIRS / AIRSIM_LIBS directly to override the defaults)
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

if(AIRSIM_INCLUDE_DIRS AND AIRSIM_LIBS)
    message(STATUS "AirSim integration enabled.")
    message(STATUS "  Includes: ${AIRSIM_INCLUDE_DIRS}")
    message(STATUS "  Libs:     ${AIRSIM_LIBS}")

    # -- PathRecorder: connects to AirSim, lets user fly, records trajectory --
    add_executable(PathRecorder main_recorder.cpp)
    target_include_directories(PathRecorder PRIVATE ${AIRSIM_INCLUDE_DIRS})
    target_link_libraries(PathRecorder PRIVATE
        ${OpenCV_LIBS}
        ${AIRSIM_LIBS}
    )

    # -- PathPlayer: replays a trajectory with the odometry pipeline running --
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
    )
    set_target_properties(PathPlayer PROPERTIES
        INSTALL_RPATH "${ONNXRUNTIME_DIR}/lib:${CMAKE_CURRENT_SOURCE_DIR}/cuda12_libs"
        BUILD_WITH_INSTALL_RPATH TRUE)
else()
    message(STATUS "AirSim not configured -- skipping PathRecorder and PathPlayer.")
    message(STATUS "  Pass -DAIRSIM_ROOT=<AirSim repo root> (or set AIRSIM_INCLUDE_DIRS / AIRSIM_LIBS) to build them.")
endif()
