#include "SuperPointDetector.h"
#include "../utils/FeatureUtils.h"
#include <iostream>
#include <filesystem>
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
    // The dynamic-shape ScatterND lowering used by remove_borders (see the
    // node_ScatterND_* nodes) appears to trip over ORT's cross-call memory
    // arena/pattern reuse: observed in practice as a perfectly alternating
    // pass/fail pattern once one Run() call's keypoint count differs enough
    // from a cached buffer plan sized by an earlier call. Disabling both
    // forces a fresh allocation per Run() instead of reusing a plan/arena
    // sized for a different shape.
    session_options.DisableMemPattern();
    session_options.DisableCpuMemArena();

    // Hardware cascade: CUDA -> CPU. ONNX Runtime has no native OpenCL EP,
    // so OPENCL collapses straight to CPU.
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

    // Model path is resolved against the process CWD; the binary expects
    // a "models/" directory next to it (build/models/superpoint.onnx).
    // Declared outside the try so the handler can report it.
    const std::string model_path = "models/superpoint.onnx";
    try {
        session = std::make_unique<Ort::Session>(*env, model_path.c_str(), session_options);
    } catch (const Ort::Exception& e) {
        // Session construction fails for two very different reasons: the model
        // file is genuinely missing, or the execution provider (CUDA) failed to
        // initialise. Reporting only the path sends you hunting for a file that
        // is sitting right there, so distinguish the two and always surface
        // ORT's own message.
        const bool missing = !std::filesystem::exists(model_path);
        std::cerr << "[SuperPoint] CRITICAL ERROR: Ort::Session failed for '" << model_path << "'.\n";
        if (missing) {
            std::cerr << "  The file does not exist relative to the CWD ("
                      << std::filesystem::current_path().string() << ").\n"
                         "  Model paths resolve against the process CWD -- run from build/.\n";
        } else {
            std::cerr << "  The model file EXISTS, so this is an execution-provider failure,\n"
                         "  not a missing model. If backend: CUDA, retry with backend: CPU.\n";
        }
        std::cerr << "  ONNX Runtime says: " << e.what() << std::endl;
        throw;
    }
}

void SuperPointDetector::detect(DeviceBuffer& image, std::vector<cv::KeyPoint>& out_keypoints, DeviceBuffer& out_descriptors) {
    if (!session) {
        initializeSession();
    }

    // Preprocess: SuperPoint wants float32 grayscale, shape [1, 1, H, W].
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

    // Inference on the whole image; backend depends on the selected EP above.
    std::vector<cv::KeyPoint> raw_keypoints;
    cv::Mat raw_descriptors;

    try {
        auto output_tensors = session->Run(Ort::RunOptions{nullptr}, input_names, input_tensors.data(), 1, output_names, 3);

        // Keypoints: [N, 2]. Some exports emit [1, N, 2] with a batch dim.
        float* kpts_ptr = output_tensors[0].GetTensorMutableData<float>();
        auto kpts_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
        int num_kpts = kpts_shape[0];
        if (kpts_shape.size() == 3) num_kpts = kpts_shape[1];

        // Scores: [N].
        float* scores_ptr = output_tensors[1].GetTensorMutableData<float>();

        // Descriptors: [N, 256]. Cloned because the Ort::Value memory dies
        // with output_tensors at the end of the try block.
        float* desc_ptr = output_tensors[2].GetTensorMutableData<float>();
        raw_descriptors = cv::Mat(num_kpts, 256, CV_32F, desc_ptr).clone();

        raw_keypoints.reserve(num_kpts);
        for (int i = 0; i < num_kpts; ++i) {
            float x = kpts_ptr[i * 2 + 0];
            float y = kpts_ptr[i * 2 + 1];
            float score = scores_ptr[i];
            // Size 8 px is the default neighbourhood reported back to OpenCV;
            // SuperPoint itself does not produce a scale per keypoint.
            raw_keypoints.emplace_back(cv::Point2f(x, y), 8.0f, -1, score);
        }
    } catch (const Ort::Exception& e) {
        std::cerr << "[SuperPoint] Inference Error: " << e.what() << std::endl;
        // A kernel failure mid-Run() can leave the session's internal
        // execution state (arena buffers, cached intermediates) corrupted:
        // observed in practice as every subsequent frame reproducing the
        // exact same error regardless of content, permanently, until the
        // process restarts. Drop the session so the next detect() call
        // lazily rebuilds a fresh one instead of reusing a wedged one.
        session.reset();
        return;
    }

    // Optional bucketing pass: prunes the dense SuperPoint output to a
    // spatially-even grid before handing it to the matcher.
    if (config.bucketing_params.enabled && !raw_keypoints.empty()) {
        std::vector<cv::KeyPoint> bucketed_kpts;
        cv::Mat bucketed_desc;

        FeatureUtils::filterByGrid(
            raw_keypoints, raw_descriptors,
            bucketed_kpts, bucketed_desc,
            width, height, config.bucketing_params
        );

        out_keypoints = bucketed_kpts;
        out_descriptors = DeviceBuffer(bucketed_desc);
    } else {
        out_keypoints = raw_keypoints;
        out_descriptors = DeviceBuffer(raw_descriptors);
    }
}
