#include "ALIKEDDetector.h"
#include "../utils/FeatureUtils.h"
#include <opencv2/imgproc.hpp>
#include <iostream>

ALIKEDDetector::ALIKEDDetector(const OdometryConfig& cfg) 
    : config(cfg), memory_info_cpu(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)) 
{
    // Extract parameters with safe fallbacks
    max_keypoints = config.detector_params.count("max_keypoints") ? 
                    (int)config.detector_params.at("max_keypoints") : 2000;
                    
    detection_threshold = config.detector_params.count("detection_threshold") ? 
                          config.detector_params.at("detection_threshold") : 0.2f;
                          
    nms_radius = config.detector_params.count("nms_radius") ? 
                 (int)config.detector_params.at("nms_radius") : 2;

    env = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_ERROR, "ALIKED");
}

void ALIKEDDetector::initializeSession() {
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(config.num_threads);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    // --- ONNX Execution Provider Fallback Logic ---
    if (config.backend == ComputeBackend::CUDA) {
        try {
            // Leverage ONNX Runtime's independent CUDA engine as per CMake configuration
            OrtCUDAProviderOptions cuda_options;
            cuda_options.device_id = 0;
            session_options.AppendExecutionProvider_CUDA(cuda_options);
            std::cout << "[ALIKEDDetector] ONNX Runtime Execution Provider: CUDA" << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "[ALIKEDDetector] WARNING: Failed to append CUDA execution provider. Falling back to CPU." << std::endl;
        }
    } else if (config.backend == ComputeBackend::OPENCL) {
        std::cerr << "[ALIKEDDetector] WARNING: ONNX OpenCL EP not configured. Falling back to CPU execution." << std::endl;
    }

    #ifdef _WIN32
        const wchar_t* model_path = L"models/aliked-n16.onnx";
    #else
        const char* model_path = "models/aliked-n16.onnx";
    #endif

    session = std::make_unique<Ort::Session>(*env, model_path, session_options);
}

void ALIKEDDetector::preprocess(const cv::Mat& image, std::vector<float>& input_tensor_values, std::vector<int64_t>& input_shape) {
    cv::Mat rgb_img;
    if (image.channels() == 1) {
        cv::cvtColor(image, rgb_img, cv::COLOR_GRAY2RGB);
    } else {
        cv::cvtColor(image, rgb_img, cv::COLOR_BGR2RGB);
    }
    
    rgb_img.convertTo(rgb_img, CV_32F, 1.0 / 255.0);

    input_shape = {1, 3, image.rows, image.cols};
    size_t image_size = image.rows * image.cols;
    input_tensor_values.resize(3 * image_size);

    std::vector<cv::Mat> chw(3);
    for (int i = 0; i < 3; ++i) {
        chw[i] = cv::Mat(image.rows, image.cols, CV_32FC1, input_tensor_values.data() + i * image_size);
    }
    cv::split(rgb_img, chw);
}

void ALIKEDDetector::detect(DeviceBuffer& image, std::vector<cv::KeyPoint>& out_keypoints, DeviceBuffer& out_descriptors) {
    if (!session) {
        initializeSession();
    }

    cv::Mat cpu_img;

    // --- Hardware Memory Fallback Logic ---
    switch (config.backend) {
        case ComputeBackend::CUDA: {
            #ifdef HAS_CUDA
            // If OpenCV CUDA is active, ensure we download it properly before feeding to ONNX
            cpu_img = image.getAsCPU(); 
            #else
            // Fallback for when OpenCV CUDA is disabled but ONNX CUDA is running
            cpu_img = image.getAsCPU(); 
            #endif
            break;
        }
        case ComputeBackend::OPENCL: {
            // Passing OpenCL UMat memory triggers the Transparent API
            cv::UMat u_img = image.getAsOpenCL();
            cpu_img = u_img.getMat(cv::ACCESS_READ);
            break;
        }
        case ComputeBackend::CPU:
        default: {
            cpu_img = image.getAsCPU();
            break;
        }
    }

    if (cpu_img.empty()) return;

    // --- Inference ---
    std::vector<float> input_tensor_values;
    std::vector<int64_t> input_shape;
    preprocess(cpu_img, input_tensor_values, input_shape);

    const char* input_names[] = {"image"};
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info_cpu, input_tensor_values.data(), input_tensor_values.size(), input_shape.data(), input_shape.size()
    );

    const char* output_names[] = {"keypoints", "descriptors", "scores"};

    auto output_tensors = session->Run(
        Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 3
    );

    // --- Post-Processing ---
    float* kpts_ptr = output_tensors[0].GetTensorMutableData<float>();
    float* desc_ptr = output_tensors[1].GetTensorMutableData<float>();
    float* scores_ptr = output_tensors[2].GetTensorMutableData<float>();

    auto kpts_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
    int num_detected_kpts = kpts_shape[1]; 
    int desc_dim = output_tensors[1].GetTensorTypeAndShapeInfo().GetShape()[2]; 

    out_keypoints.clear();
    out_keypoints.reserve(num_detected_kpts);
    
    cv::Mat cpu_desc(num_detected_kpts, desc_dim, CV_32F);

    int count = 0;
    for (int i = 0; i < num_detected_kpts; ++i) {
        if (scores_ptr[i] < detection_threshold) continue;
        if (count >= max_keypoints) break;

        float x = kpts_ptr[i * 2];
        float y = kpts_ptr[i * 2 + 1];

        out_keypoints.emplace_back(x, y, 1.0f, -1, scores_ptr[i]);
        
        std::memcpy(cpu_desc.ptr<float>(count), desc_ptr + (i * desc_dim), desc_dim * sizeof(float));
        count++;
    }

    cpu_desc = cpu_desc.rowRange(0, count);
    
    out_descriptors.getAsCPU() = cpu_desc.clone(); 
}
