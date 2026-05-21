cmake_minimum_required(VERSION 3.18) 
project(OdometryKittiTest LANGUAGES CXX) 

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

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
)

# --- BURN RPATHS INTO BINARY ---
# We use $ORIGIN/.. to make the path relative to your build/ directory,
# ensuring it always finds the cuda12_libs folder in your project root!
set_target_properties(KittiEvaluator PROPERTIES INSTALL_RPATH "${ONNXRUNTIME_DIR}/lib:${CMAKE_CURRENT_SOURCE_DIR}/cuda12_libs")
set_target_properties(KittiEvaluator PROPERTIES BUILD_WITH_INSTALL_RPATH TRUE)
