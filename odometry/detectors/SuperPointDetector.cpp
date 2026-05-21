#include "SuperPointDetector.h"
#include "../utils/FeatureUtils.h"
#include <iostream>
#include <opencv2/imgproc.hpp>

SuperPointDetector::SuperPointDetector(const OdometryConfig& cfg) 
    : config(cfg), 
      memory_info_cpu(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)) 
{
    env = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_ERROR, "SuperPoint");
}

void SuperPointDetector::initializeSession() {
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(config.num_threads);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    session_options.SetLogSeverityLevel(3);

    // ==========================================
    // 1. HARDWARE CAPABILITY RESOLVER
    // Strict Cascade: CUDA -> CPU (ONNX does not natively support OpenCL)
    // ==========================================
    ComputeBackend target_backend = config.backend;

    if (target_backend == ComputeBackend::OPENCL) {
        static bool warned_ocl = false;
        if (!warned_ocl) {
            std::cerr << "[SuperPoint] ONNX Runtime does not natively support OpenCL. Cascading to CPU..." << std::endl;
            warned_ocl = true;
        }
        target_backend = ComputeBackend::CPU;
    }

    if (target_backend == ComputeBackend::CUDA) {
        try {
            OrtCUDAProviderOptions cuda_options;
            cuda_options.device_id = 0; 
            session_options.AppendExecutionProvider_CUDA(cuda_options);
            std::cout << "[SuperPoint] Backend: CUDA Execution Provider Enabled." << std::endl;
        } catch (const Ort::Exception& e) {
            std::cerr << "[SuperPoint] CUDA Init Failed: " << e.what() << std::endl;
            std::cerr << "[SuperPoint] Cascading to CPU Execution Provider." << std::endl;
            target_backend = ComputeBackend::CPU;
        }
    }

    if (target_backend == ComputeBackend::CPU) {
        std::cout << "[SuperPoint] Backend: CPU Execution Provider." << std::endl;
    }

    try {
        // Ensure you have this model exported and placed alongside lightglue!
        std::string model_path = "models/superpoint.onnx"; 
        session = std::make_unique<Ort::Session>(*env, model_path.c_str(), session_options);
    } catch (const Ort::Exception& e) {
        std::cerr << "[SuperPoint] CRITICAL ERROR: Could not load models/superpoint.onnx" << std::endl;
        throw;
    }
}

void SuperPointDetector::detect(DeviceBuffer& image, std::vector<cv::KeyPoint>& out_keypoints, DeviceBuffer& out_descriptors) {
    if (!session) {
        initializeSession();
    }

    // 1. Preprocess Image for SuperPoint (Float32, Grayscale, 1x1xHxW)
    cv::Mat cpu_img = image.getAsCPU();
    cv::Mat gray_img;
    if (cpu_img.channels() == 3) {
        cv::cvtColor(cpu_img, gray_img, cv::COLOR_BGR2GRAY);
    } else {
        gray_img = cpu_img;
    }

    cv::Mat float_img;
    gray_img.convertTo(float_img, CV_32F, 1.0 / 255.0);

    int height = float_img.rows;
    int width = float_img.cols;
    std::vector<int64_t> input_shape = {1, 1, height, width};

    std::vector<Ort::Value> input_tensors;
    input_tensors.push_back(Ort::Value::CreateTensor<float>(
        memory_info_cpu, (float*)float_img.data, float_img.total(), input_shape.data(), input_shape.size()
    ));

    const char* input_names[] = {"image"};
    const char* output_names[] = {"keypoints", "scores", "descriptors"};

    // 2. Execute GPU Inference (Whole Image)
    std::vector<cv::KeyPoint> raw_keypoints;
    cv::Mat raw_descriptors;

    try {
        auto output_tensors = session->Run(Ort::RunOptions{nullptr}, input_names, input_tensors.data(), 1, output_names, 3);

        // Extract Keypoints [N, 2]
        float* kpts_ptr = output_tensors[0].GetTensorMutableData<float>();
        auto kpts_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
        int num_kpts = kpts_shape[0]; // Assuming standard SuperPoint ONNX output shape [N, 2] or [1, N, 2]
        if (kpts_shape.size() == 3) num_kpts = kpts_shape[1]; 

        // Extract Scores [N]
        float* scores_ptr = output_tensors[1].GetTensorMutableData<float>();

        // Extract Descriptors [N, 256]
        float* desc_ptr = output_tensors[2].GetTensorMutableData<float>();
        raw_descriptors = cv::Mat(num_kpts, 256, CV_32F, desc_ptr).clone(); // Clone to preserve after Ort::Value dies

        raw_keypoints.reserve(num_kpts);
        for (int i = 0; i < num_kpts; ++i) {
            float x = kpts_ptr[i * 2 + 0];
            float y = kpts_ptr[i * 2 + 1];
            float score = scores_ptr[i];
            raw_keypoints.emplace_back(cv::Point2f(x, y), 8.0f, -1, score); // Default size 8
        }
    } catch (const Ort::Exception& e) {
        std::cerr << "[SuperPoint] Inference Error: " << e.what() << std::endl;
        return;
    }

    // ==========================================
    // 3. POST-DETECTION LOGIC (Bucketing)
    // ==========================================
    if (config.bucketing_params.enabled && !raw_keypoints.empty()) {
        std::vector<cv::KeyPoint> bucketed_kpts;
        cv::Mat bucketed_desc;
        
        // Filter the dense GPU outputs into an even spatial grid
        FeatureUtils::filterByGrid(
            raw_keypoints, raw_descriptors, 
            bucketed_kpts, bucketed_desc, 
            width, height, config.bucketing_params
        );

        out_keypoints = bucketed_kpts;
        out_descriptors = DeviceBuffer(bucketed_desc);
    } else {
        // Bypass filtering
        out_keypoints = raw_keypoints;
        out_descriptors = DeviceBuffer(raw_descriptors);
    }
}
