#include "ORBDetector.h"
#include "../utils/FeatureUtils.h"
#include <iostream>
#include <opencv2/imgproc.hpp>

// OpenCL T-API availability check.
#include <opencv2/core/ocl.hpp>

#ifdef HAS_CUDA
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudafeatures2d.hpp>
#endif

ORBDetector::ORBDetector(const OdometryConfig& cfg) : config(cfg) {
    // Params: pull from config with literal fallbacks. Fallbacks for the
    // optional knobs match cv::ORB::create's own defaults so that omitting
    // them in YAML produces identical behaviour to the bare constructor.
    const auto& p = config.detector_params;
    auto getI = [&](const char* key, int def) {
        return p.count(key) ? (int)p.at(key) : def;
    };
    auto getF = [&](const char* key, float def) {
        return p.count(key) ? p.at(key) : def;
    };

    num_features   = getI("nfeatures",      500);
    scale_factor   = getF("scale_factor",   1.2f);
    nlevels        = getI("nlevels",        8);
    edge_threshold = getI("edge_threshold", 31);
    first_level    = getI("first_level",    0);
    wta_k          = getI("wta_k",          2);
    patch_size     = getI("patch_size",     31);
    fast_threshold = getI("fast_threshold", 20);

    std::cout << "[ORBDetector] nfeatures=" << num_features
              << " scaleFactor=" << scale_factor
              << " nLevels=" << nlevels
              << " edgeThreshold=" << edge_threshold
              << " firstLevel=" << first_level
              << " WTA_K=" << wta_k
              << " patchSize=" << patch_size
              << " fastThreshold=" << fast_threshold << std::endl;

    // CPU/OpenCL object is always built; it is the fallback target of the
    // CUDA -> OpenCL -> CPU cascade, and the whole-image (non-bucketed) path.
    orb_cpu = cv::ORB::create(num_features, scale_factor, nlevels,
                              edge_threshold, first_level, wta_k,
                              cv::ORB::HARRIS_SCORE, patch_size, fast_threshold);

    // CUDA object: only when compiled in and explicitly requested.
#ifdef HAS_CUDA
    if (config.backend == ComputeBackend::CUDA) {
        orb_cuda = cv::cuda::ORB::create(num_features, scale_factor, nlevels,
                                         edge_threshold, first_level, wta_k,
                                         cv::ORB::HARRIS_SCORE, patch_size, fast_threshold);
    }
#endif
}

void ORBDetector::detect(DeviceBuffer& image,
                         std::vector<cv::KeyPoint>& out_keypoints,
                         DeviceBuffer& out_descriptors) {

    // Hardware cascade: CUDA -> OpenCL -> CPU.
    ComputeBackend target_backend = config.backend;

    // Bucketing override: per-tile ORB runs in CPU threads (see the CPU
    // branch below), same reasoning as SIFTDetector -- round-tripping tiles
    // to a GPU would dominate compute time.
    if (config.bucketing_params.enabled && target_backend != ComputeBackend::CPU) {
        static bool warned_bucketing = false;
        if (!warned_bucketing) {
            std::cerr << "[ORBDetector] True spatial bucketing requested. Bypassing GPU to prevent PCIe bottlenecks. Forcing CPU Multithreading..." << std::endl;
            warned_bucketing = true;
        }
        target_backend = ComputeBackend::CPU;
    }

    switch (target_backend) {

        case ComputeBackend::CUDA: {
#ifdef HAS_CUDA
            // Image is already on device; getCUDAPointer hands back the GpuMat.
            cv::cuda::GpuMat* d_img = static_cast<cv::cuda::GpuMat*>(image.getCUDAPointer());

            // ORB needs grayscale; convert if input is BGR.
            cv::cuda::GpuMat d_gray;
            if (d_img->channels() == 3) {
                cv::cuda::cvtColor(*d_img, d_gray, cv::COLOR_BGR2GRAY);
            } else {
                d_gray = *d_img;
            }

            cv::cuda::GpuMat d_descriptors;
            orb_cuda->detectAndComputeAsync(d_gray, cv::cuda::GpuMat(), out_keypoints, d_descriptors);

            out_descriptors = DeviceBuffer(d_descriptors);
            break;
#else
            std::cerr << "[ORBDetector] WARNING: CUDA requested but not compiled. Cascading to OpenCL." << std::endl;
            // Intentional fallthrough: next case is OPENCL, which itself
            // cascades to CPU if OpenCL is unavailable at runtime.
#endif
        }

        case ComputeBackend::OPENCL: {
            // Runtime check: if OpenCL is missing, the T-API transparently
            // executes the same calls on the host as plain cv::Mat ops.
            if (!cv::ocl::haveOpenCL()) {
                std::cerr << "[ORBDetector] WARNING: OpenCL not available. Falling back to CPU." << std::endl;
            }

            // getAsOpenCL copies to cv::UMat; OpenCV uses the OpenCL kernel
            // path whenever a UMat is passed in.
            cv::UMat u_img = image.getAsOpenCL();
            cv::UMat u_gray;

            if (u_img.channels() == 3) {
                cv::cvtColor(u_img, u_gray, cv::COLOR_BGR2GRAY);
            } else {
                u_gray = u_img;
            }

            cv::UMat u_descriptors;
            orb_cpu->detectAndCompute(u_gray, cv::noArray(), out_keypoints, u_descriptors);

            out_descriptors = DeviceBuffer(u_descriptors);
            break;
        }

        case ComputeBackend::CPU:
        default: {
            cv::Mat cpu_img = image.getAsCPU();
            cv::Mat gray_img;

            if (cpu_img.channels() == 3) {
                cv::cvtColor(cpu_img, gray_img, cv::COLOR_BGR2GRAY);
            } else {
                gray_img = cpu_img;
            }

            cv::Mat cpu_descriptors;

            if (config.bucketing_params.enabled) {
                // Per-tile ORB, same mechanism SIFTDetector uses: each grid
                // cell is detected independently and capped at
                // max_features_per_bucket, so a low-texture cell's real (but
                // globally-outranked) corners survive instead of losing to a
                // high-contrast cell in a single whole-image top-nfeatures
                // selection.
                auto builder = [this]() {
                    return cv::ORB::create(num_features, scale_factor, nlevels,
                                            edge_threshold, first_level, wta_k,
                                            cv::ORB::HARRIS_SCORE, patch_size, fast_threshold);
                };
                FeatureUtils::detectWithGridCPU(builder, gray_img, out_keypoints, cpu_descriptors,
                                                 config.bucketing_params, config.num_threads);
            } else {
                orb_cpu->detectAndCompute(gray_img, cv::noArray(), out_keypoints, cpu_descriptors);
            }

            out_descriptors = DeviceBuffer(cpu_descriptors);
            break;
        }
    }
}
