cmake_minimum_required(VERSION 3.15)
project(OdometryKittiTest LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- FIND OPENCV 4.8.0 ---
# Since you have a custom build, you might need to specify the path when running cmake:
# cmake -DOpenCV_DIR=/path/to/your/opencv/build ..
find_package(OpenCV 4.8.0 REQUIRED)
message(STATUS "Found OpenCV: ${OpenCV_VERSION} at ${OpenCV_DIR}")

# --- CUDA CHECK ---
# For this testing branch, we are intentionally disabling CUDA compilation
# to avoid linking errors with missing opencv_contrib modules (cudafeatures2d).
# The pipeline will gracefully route to the CPU/OpenCL fallback.

# list(FIND OpenCV_INCLUDE_DIRS "opencv2/core/cuda.hpp" CUDA_FOUND_IDX)
# if(NOT CUDA_FOUND_IDX EQUAL -1 OR EXISTS "${OpenCV_INCLUDE_DIRS}/opencv2/core/cuda.hpp")
#     message(STATUS "OpenCV CUDA modules detected.")
#     add_compile_definitions(HAS_CUDA)
# else()
#     message(WARNING "OpenCV CUDA not found. Defaulting to CPU/OpenCL.")
# endif()

message(STATUS "Testing Branch: Forcing CPU/OpenCL mode. CUDA compilation disabled.")

# --- GATHER ODOMETRY SOURCE FILES ---
file(GLOB_RECURSE SRC_ODOMETRY "odometry/*.cpp")

# --- BUILD EXECUTABLE ---
add_executable(KittiEvaluator 
    main_test.cpp
    ${SRC_ODOMETRY}
)

target_link_libraries(KittiEvaluator PRIVATE ${OpenCV_LIBS})
