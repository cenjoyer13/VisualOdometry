#define _USE_MATH_DEFINES  // Must be before any header that transitively pulls <cmath>.
#include "LightGlueMatcher.h"
#include <iostream>
#include <cmath>
#include <algorithm>
#include <string>

LightGlueMatcher::LightGlueMatcher(const OdometryConfig& cfg)
    : config(cfg),
      memory_info_cpu(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)),
      is_sift(false)
{
    env = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_ERROR, "LightGlue");
}

void LightGlueMatcher::initializeSession(int desc_dim) {
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(config.num_threads);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    // Severity 3 silences ORT info/warning logs; only errors and fatals.
    session_options.SetLogSeverityLevel(3);

    // Execution provider selection. ONNX RT has no native OpenCL EP, so the
    // OPENCL branch collapses to CPU.
    switch (config.backend) {
        case ComputeBackend::CUDA: {
            try {
                OrtCUDAProviderOptionsV2* cuda_options = nullptr;
                Ort::ThrowOnError(Ort::GetApi().CreateCUDAProviderOptions(&cuda_options));

                std::vector<const char*> keys = {"device_id"};
                std::vector<const char*> values = {"0"};

                Ort::ThrowOnError(Ort::GetApi().UpdateCUDAProviderOptions(cuda_options, keys.data(), values.data(), keys.size()));

                session_options.AppendExecutionProvider_CUDA_V2(*cuda_options);

                Ort::GetApi().ReleaseCUDAProviderOptions(cuda_options);

                std::cout << "[LightGlue] Backend: CUDA Execution Provider Enabled (V2 API)." << std::endl;
            } catch (const Ort::Exception& e) {
                std::cerr << "[LightGlue] CUDA Init Failed: " << e.what() << std::endl;
                std::cerr << "[LightGlue] Falling back to CPU Execution Provider." << std::endl;
            }
            break;
        }
        case ComputeBackend::OPENCL: {
            std::cout << "[LightGlue] Backend: OPENCL requested. ONNX Runtime neural networks do not natively support OpenCL. Safely falling back to CPU." << std::endl;
            break;
        }
        case ComputeBackend::CPU:
        default: {
            std::cout << "[LightGlue] Backend: CPU Execution Provider." << std::endl;
            break;
        }
    }

    std::string model_path;

    if (desc_dim == 128) {
        is_sift = true;
        model_path = "models/sift_lightglue.onnx";
        std::cout << "[LightGlue] Detected 128D Descriptors. Loading SIFT Model (8 Inputs)." << std::endl;
    } else if (desc_dim == 256) {
        is_sift = false;
        model_path = "models/superpoint_lightglue.onnx";
        std::cout << "[LightGlue] Detected 256D Descriptors. Loading SuperPoint Model (4 Inputs)." << std::endl;
    } else {
        throw std::runtime_error("[LightGlue] Unsupported descriptor dimension. Expected 128 or 256.");
    }

    try {
        #ifdef _WIN32
            std::wstring wpath(model_path.begin(), model_path.end());
            session = std::make_unique<Ort::Session>(*env, wpath.c_str(), session_options);
        #else
            session = std::make_unique<Ort::Session>(*env, model_path.c_str(), session_options);
        #endif
        std::cout << "[LightGlue] ONNX Session booted successfully." << std::endl;
    } catch (const Ort::Exception& e) {
        std::cerr << "[LightGlue] CRITICAL ONNX ERROR: " << e.what() << std::endl;
        throw;
    }
}

void LightGlueMatcher::convertToRootSift(cv::Mat& desc) {
    if (desc.type() != CV_32F) desc.convertTo(desc, CV_32F);

    for (int i = 0; i < desc.rows; ++i) {
        cv::Mat row = desc.row(i);
        cv::normalize(row, row, 1.0, 0.0, cv::NORM_L1);
    }
    cv::sqrt(desc, desc);
}

std::vector<cv::DMatch> LightGlueMatcher::match(DeviceBuffer& desc_old, 
                                                DeviceBuffer& desc_new,
                                                const std::vector<cv::KeyPoint>& kp_old,
                                                const std::vector<cv::KeyPoint>& kp_new) 
{
    if (kp_old.empty() || kp_new.empty()) return {};

    // Stage descriptors on the host: ONNX RT's default Run path takes host
    // pointers and handles its own device transfers when running on CUDA.
    cv::Mat d0 = desc_old.getAsCPU().clone();
    cv::Mat d1 = desc_new.getAsCPU().clone();

    if (d0.type() != CV_32F) d0.convertTo(d0, CV_32F);
    if (d1.type() != CV_32F) d1.convertTo(d1, CV_32F);

    // Session is built on the first frame, when desc_dim is known.
    if (!session) {
        initializeSession(d0.cols);
    }

    // Resolve the confidence-score output name. Different LightGlue ONNX
    // exports name it differently ("scores", "mscores", "matching_scores"),
    // so the search is by substring and cached for subsequent frames.
    static std::string score_node_name = "";
    static bool node_checked = false;

    if (!node_checked) {
        Ort::AllocatorWithDefaultOptions allocator;
        for (size_t i = 0; i < session->GetOutputCount(); i++) {
            std::string name = session->GetOutputNameAllocated(i, allocator).get();
            if (name.find("score") != std::string::npos) {
                score_node_name = name;
            }
        }
        node_checked = true;
        if (!score_node_name.empty()) {
            std::cout << "[LightGlue] Dynamically locked confidence score node: '" << score_node_name << "'" << std::endl;
        }
    }

    // RootSIFT transform is only required for the 128D SIFT branch.
    if (is_sift) {
        convertToRootSift(d0);
        convertToRootSift(d1);
    }

    int N = kp_old.size();
    int M = kp_new.size();
    int desc_dim = d0.cols;

    std::vector<float> kp0_data(N * 2), kp1_data(M * 2);
    std::vector<float> scales0_data, scales1_data;
    std::vector<float> oris0_data, oris1_data;

    const float deg_to_rad = static_cast<float>(M_PI / 180.0);

    for (int i = 0; i < N; ++i) {
        kp0_data[i * 2 + 0] = kp_old[i].pt.x;
        kp0_data[i * 2 + 1] = kp_old[i].pt.y;
    }
    for (int i = 0; i < M; ++i) {
        kp1_data[i * 2 + 0] = kp_new[i].pt.x;
        kp1_data[i * 2 + 1] = kp_new[i].pt.y;
    }

    if (is_sift) {
        scales0_data.resize(N); scales1_data.resize(M);
        oris0_data.resize(N);   oris1_data.resize(M);
        
        for (int i = 0; i < N; ++i) {
            scales0_data[i] = kp_old[i].size;
            oris0_data[i]   = kp_old[i].angle * deg_to_rad;
        }
        for (int i = 0; i < M; ++i) {
            scales1_data[i] = kp_new[i].size;
            oris1_data[i]   = kp_new[i].angle * deg_to_rad;
        }
    }

    std::vector<int64_t> kpts0_shape = {1, N, 2};
    std::vector<int64_t> kpts1_shape = {1, M, 2};
    std::vector<int64_t> desc0_shape = {1, N, desc_dim};
    std::vector<int64_t> desc1_shape = {1, M, desc_dim};
    std::vector<int64_t> scale0_shape = {1, N};
    std::vector<int64_t> scale1_shape = {1, M};
    std::vector<int64_t> ori0_shape   = {1, N};
    std::vector<int64_t> ori1_shape   = {1, M};

    std::vector<Ort::Value> input_tensors;
    std::vector<const char*> input_names;

    input_tensors.push_back(Ort::Value::CreateTensor<float>(memory_info_cpu, kp0_data.data(), kp0_data.size(), kpts0_shape.data(), kpts0_shape.size()));
    input_tensors.push_back(Ort::Value::CreateTensor<float>(memory_info_cpu, kp1_data.data(), kp1_data.size(), kpts1_shape.data(), kpts1_shape.size()));
    input_tensors.push_back(Ort::Value::CreateTensor<float>(memory_info_cpu, (float*)d0.data, d0.total(), desc0_shape.data(), desc0_shape.size()));
    input_tensors.push_back(Ort::Value::CreateTensor<float>(memory_info_cpu, (float*)d1.data, d1.total(), desc1_shape.data(), desc1_shape.size()));
    
    input_names = {"kpts0", "kpts1", "desc0", "desc1"};

    if (is_sift) {
        input_tensors.push_back(Ort::Value::CreateTensor<float>(memory_info_cpu, scales0_data.data(), scales0_data.size(), scale0_shape.data(), scale0_shape.size()));
        input_tensors.push_back(Ort::Value::CreateTensor<float>(memory_info_cpu, scales1_data.data(), scales1_data.size(), scale1_shape.data(), scale1_shape.size()));
        input_tensors.push_back(Ort::Value::CreateTensor<float>(memory_info_cpu, oris0_data.data(), oris0_data.size(), ori0_shape.data(), ori0_shape.size()));
        input_tensors.push_back(Ort::Value::CreateTensor<float>(memory_info_cpu, oris1_data.data(), oris1_data.size(), ori1_shape.data(), ori1_shape.size()));
        
        input_names.push_back("scales0");
        input_names.push_back("scales1");
        input_names.push_back("oris0");
        input_names.push_back("oris1");
    }

    // Request the scores output only if the model exposes one.
    std::vector<const char*> output_names = {"matches"};
    if (!score_node_name.empty()) {
        output_names.push_back(score_node_name.c_str());
    }

    try {
        auto output_tensors = session->Run(Ort::RunOptions{nullptr}, input_names.data(), input_tensors.data(), input_tensors.size(), output_names.data(), output_names.size());

        auto* matches_out = output_tensors[0].GetTensorMutableData<int64_t>();
        auto match_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();

        float* scores_out = nullptr;
        if (!score_node_name.empty() && output_tensors.size() > 1) {
            scores_out = output_tensors[1].GetTensorMutableData<float>();
        }

        int num_matches = match_shape[0];
        std::vector<cv::DMatch> good_matches;
        good_matches.reserve(num_matches);

        for (int i = 0; i < num_matches; ++i) {
            int idx0 = static_cast<int>(matches_out[i * 2 + 0]);
            int idx1 = static_cast<int>(matches_out[i * 2 + 1]);

            float distance = 0.0f;
            if (scores_out) {
                // cv::DMatch::distance is treated as "lower is better" by the
                // rest of the pipeline; invert the LightGlue confidence so
                // strongest matches sort first.
                distance = 1.0f - scores_out[i];
            }

            good_matches.emplace_back(idx0, idx1, distance);
        }

        std::stable_sort(good_matches.begin(), good_matches.end(), [](const cv::DMatch& a, const cv::DMatch& b) {
            return a.distance < b.distance;
        });

        return good_matches;

    } catch (const Ort::Exception& e) {
        std::cerr << "[LightGlue Matcher Error]: " << e.what() << std::endl;
        return {};
    }
}
