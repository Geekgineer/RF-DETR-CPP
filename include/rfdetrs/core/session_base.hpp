// SPDX-FileCopyrightText: 2024 Geekgineer
// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

// ============================================================================
// RFDETRs-CPP — ONNX Runtime session base
// ============================================================================
// OrtSessionBase: handles session creation, EP selection (CPU / CUDA /
// TensorRT), input/output node enumeration, and the RF-DETR-specific
// shape-based identification of box vs. logit output tensors.

#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>

#include <algorithm>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "rfdetrs/core/onnx_metadata.hpp"
#include "rfdetrs/core/utils.hpp"

namespace rfdetrs {

// ============================================================================
// BackendType
// ============================================================================

enum class BackendType {
    CPU,       ///< Default CPU execution provider
    CUDA,      ///< NVIDIA CUDA EP (falls back to CPU if unavailable)
    TensorRT,  ///< NVIDIA TensorRT EP (falls back CUDA → CPU if unavailable)
};

// ============================================================================
// OrtSessionBase
// ============================================================================

class OrtSessionBase {
public:
    OrtSessionBase(const std::string& modelPath,
                   BackendType backend = BackendType::CPU,
                   int numThreads = 0)
        : env_(ORT_LOGGING_LEVEL_WARNING, "RFDETRs") {
        initSession(modelPath, backend, numThreads);
    }

    virtual ~OrtSessionBase() = default;

    OrtSessionBase(const OrtSessionBase&) = delete;
    OrtSessionBase& operator=(const OrtSessionBase&) = delete;
    OrtSessionBase(OrtSessionBase&&) = default;
    OrtSessionBase& operator=(OrtSessionBase&&) = default;

    [[nodiscard]] cv::Size getInputShape()   const noexcept { return inputShape_; }
    [[nodiscard]] const std::string& getDevice() const noexcept { return device_; }
    [[nodiscard]] size_t getNumInputNodes()  const noexcept { return numInputNodes_; }
    [[nodiscard]] size_t getNumOutputNodes() const noexcept { return numOutputNodes_; }
    [[nodiscard]] int getNumQueries()        const noexcept { return numQueries_; }
    [[nodiscard]] int getNumClasses()        const noexcept { return numClasses_; }

    /// Class names read from ONNX custom metadata (empty for RF-DETR exports).
    [[nodiscard]] const std::vector<std::string>& getEmbeddedClassNames() const noexcept {
        return embeddedClassNames_;
    }

protected:
    Ort::Env env_{nullptr};
    Ort::SessionOptions sessionOptions_{nullptr};
    Ort::Session session_{nullptr};

    std::vector<Ort::AllocatedStringPtr> inputNameAllocs_;
    std::vector<const char*> inputNames_;
    std::vector<Ort::AllocatedStringPtr> outputNameAllocs_;
    std::vector<const char*> outputNames_;

    size_t numInputNodes_{0};
    size_t numOutputNodes_{0};

    cv::Size inputShape_;
    std::string device_{"cpu"};

    int numQueries_{300};
    int numClasses_{90};
    int boxOutputIdx_{0};
    int logitOutputIdx_{1};
    int maskOutputIdx_{-1};      ///< Index of the "masks" output (-1 if not a seg model)
    int keypointOutputIdx_{-1};  ///< Index of the "keypoints" output (-1 if not a keypoint model)
    int maskH_{0};               ///< Mask spatial height (H/mask_downsample_ratio)
    int maskW_{0};               ///< Mask spatial width  (W/mask_downsample_ratio)
    int keypointSlots_{0};       ///< C * max(K_c) — keypoints tensor dim-2
    int keypointDim_{0};         ///< KEYPOINT_PRED_DIM (8)

    std::vector<float> blob_;
    std::vector<std::string> embeddedClassNames_;

    std::vector<Ort::Value> runInference(Ort::Value& inputTensor) {
        return session_.Run(
            Ort::RunOptions{nullptr},
            inputNames_.data(), &inputTensor, numInputNodes_,
            outputNames_.data(), numOutputNodes_);
    }

    Ort::Value createInputTensor(const std::vector<int64_t>& shape) {
        static Ort::MemoryInfo memInfo =
            Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        size_t n = utils::vectorProduct(shape);
        return Ort::Value::CreateTensor<float>(memInfo, blob_.data(), n,
                                               shape.data(), shape.size());
    }

private:
    void initSession(const std::string& modelPath, BackendType backend, int numThreads) {
        sessionOptions_ = Ort::SessionOptions();
        int threads = (numThreads > 0)
            ? numThreads
            : std::min(6, static_cast<int>(std::thread::hardware_concurrency()));
        sessionOptions_.SetIntraOpNumThreads(threads);
        // Single-model topology: disable inter-op parallelism (no benefit, adds overhead).
        sessionOptions_.SetInterOpNumThreads(1);
        sessionOptions_.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
        sessionOptions_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        auto availableProviders = Ort::GetAvailableProviders();
        auto hasProvider = [&](const char* name) {
            return std::find(availableProviders.begin(), availableProviders.end(),
                             std::string(name)) != availableProviders.end();
        };

        switch (backend) {
            case BackendType::CUDA:
                if (hasProvider("CUDAExecutionProvider")) {
                    OrtCUDAProviderOptions cudaOptions{};
                    sessionOptions_.AppendExecutionProvider_CUDA(cudaOptions);
                    device_ = "gpu:cuda";
                    std::cout << "[INFO] Backend: GPU (CUDA)\n";
                } else {
                    std::cout << "[WARNING] CUDA EP not available, falling back to CPU.\n";
                    device_ = "cpu";
                }
                break;

            case BackendType::TensorRT:
                if (hasProvider("TensorrtExecutionProvider")) {
                    OrtTensorRTProviderOptions trtOptions{};
                    sessionOptions_.AppendExecutionProvider_TensorRT(trtOptions);
                    device_ = "gpu:trt";
                    std::cout << "[INFO] Backend: GPU (TensorRT) — first run builds engine cache.\n";
                } else if (hasProvider("CUDAExecutionProvider")) {
                    std::cout << "[WARNING] TensorRT EP not available, falling back to CUDA.\n";
                    OrtCUDAProviderOptions cudaOptions{};
                    sessionOptions_.AppendExecutionProvider_CUDA(cudaOptions);
                    device_ = "gpu:cuda";
                } else {
                    std::cout << "[WARNING] TensorRT and CUDA EPs not available, falling back to CPU.\n";
                    device_ = "cpu";
                }
                break;

            case BackendType::CPU:
            default:
                device_ = "cpu";
                std::cout << "[INFO] Backend: CPU\n";
                break;
        }

#ifdef _WIN32
        std::wstring wPath(modelPath.begin(), modelPath.end());
        session_ = Ort::Session(env_, wPath.c_str(), sessionOptions_);
#else
        session_ = Ort::Session(env_, modelPath.c_str(), sessionOptions_);
#endif

        numInputNodes_  = session_.GetInputCount();
        numOutputNodes_ = session_.GetOutputCount();

        Ort::AllocatorWithDefaultOptions allocator;

        for (size_t i = 0; i < numInputNodes_; ++i) {
            auto name = session_.GetInputNameAllocated(i, allocator);
            inputNameAllocs_.push_back(std::move(name));
            inputNames_.push_back(inputNameAllocs_.back().get());
        }
        for (size_t i = 0; i < numOutputNodes_; ++i) {
            auto name = session_.GetOutputNameAllocated(i, allocator);
            outputNameAllocs_.push_back(std::move(name));
            outputNames_.push_back(outputNameAllocs_.back().get());
        }

        // Input shape [1, 3, H, W]
        auto inputTypeInfo = session_.GetInputTypeInfo(0);
        auto inputShape = inputTypeInfo.GetTensorTypeAndShapeInfo().GetShape();
        if (inputShape.size() < 4)
            throw std::runtime_error("Expected 4-D input tensor [1, 3, H, W]");
        inputShape_ = cv::Size(static_cast<int>(inputShape[3]),
                               static_cast<int>(inputShape[2]));

        // Identify box output ("dets", last-dim==4) and logit output ("labels"/"logits").
        // Strategy: name-based pass first, then shape-based fallback.
        // The "keypoints" output has last-dim==8 which would otherwise be mis-identified
        // as the logit output for models with few classes (e.g. keypoint model: 2 classes).
        boxOutputIdx_   = -1;
        logitOutputIdx_ = -1;

        // Pass 1: name-based identification
        for (size_t i = 0; i < numOutputNodes_; ++i) {
            const std::string& name = outputNames_[i];
            auto outTypeInfo = session_.GetOutputTypeInfo(i);
            auto shape = outTypeInfo.GetTensorTypeAndShapeInfo().GetShape();
            if (shape.size() < 3) continue;
            int64_t lastDim = shape.back();
            if (name == "dets" && boxOutputIdx_ == -1 && lastDim == 4) {
                boxOutputIdx_ = static_cast<int>(i);
                numQueries_   = static_cast<int>(shape[shape.size() - 2]);
            } else if ((name == "labels" || name == "logits") && logitOutputIdx_ == -1) {
                logitOutputIdx_ = static_cast<int>(i);
                numClasses_     = static_cast<int>(lastDim);
            }
        }

        // Pass 2: shape-based fallback (non-standard output names)
        for (size_t i = 0; i < numOutputNodes_; ++i) {
            const std::string& name = outputNames_[i];
            // Never mis-identify keypoints/masks outputs as boxes or logits
            if (name == "keypoints" || name == "masks") continue;
            if (static_cast<int>(i) == boxOutputIdx_ ||
                static_cast<int>(i) == logitOutputIdx_) continue;
            auto outTypeInfo = session_.GetOutputTypeInfo(i);
            auto shape = outTypeInfo.GetTensorTypeAndShapeInfo().GetShape();
            if (shape.size() < 3) continue;
            int64_t lastDim = shape.back();
            if (lastDim == 4 && boxOutputIdx_ == -1) {
                boxOutputIdx_ = static_cast<int>(i);
                numQueries_   = static_cast<int>(shape[shape.size() - 2]);
            } else if (lastDim > 4 && logitOutputIdx_ == -1) {
                logitOutputIdx_ = static_cast<int>(i);
                numClasses_     = static_cast<int>(lastDim);
            }
        }

        if (boxOutputIdx_ == -1 || logitOutputIdx_ == -1)
            throw std::runtime_error(
                "Cannot identify box/logit outputs. "
                "Expected outputs named 'dets'/'labels' or shaped [N,4]/[N,C>4].");

        // Identify optional 3rd outputs (segmentation masks or keypoints) by name.
        for (size_t i = 0; i < numOutputNodes_; ++i) {
            std::string name = outputNames_[i];
            auto outInfo = session_.GetOutputTypeInfo(i);
            auto shape   = outInfo.GetTensorTypeAndShapeInfo().GetShape();
            if (name == "masks" && maskOutputIdx_ == -1) {
                maskOutputIdx_ = static_cast<int>(i);
                if (shape.size() >= 4) {
                    maskH_ = static_cast<int>(shape[static_cast<size_t>(shape.size() - 2)]);
                    maskW_ = static_cast<int>(shape.back());
                }
            } else if (name == "keypoints" && keypointOutputIdx_ == -1) {
                keypointOutputIdx_ = static_cast<int>(i);
                if (shape.size() >= 4) {
                    keypointSlots_ = static_cast<int>(shape[static_cast<size_t>(shape.size() - 2)]);
                    keypointDim_   = static_cast<int>(shape.back());
                }
            }
        }

        blob_.resize(static_cast<size_t>(3 * inputShape_.height * inputShape_.width));

        std::cout << "[INFO] Model loaded: " << modelPath << "\n";
        std::cout << "[INFO] Input: " << inputShape_.width << "x" << inputShape_.height << "\n";
        std::cout << "[INFO] Queries: " << numQueries_ << "  Classes: " << numClasses_ << "\n";
        std::cout << "[INFO] Box output idx: " << boxOutputIdx_
                  << "  Logit output idx: " << logitOutputIdx_;
        if (maskOutputIdx_ >= 0)
            std::cout << "  Mask output idx: " << maskOutputIdx_
                      << " (" << maskH_ << "x" << maskW_ << ")";
        if (keypointOutputIdx_ >= 0)
            std::cout << "  Keypoint output idx: " << keypointOutputIdx_
                      << " (slots=" << keypointSlots_ << " dim=" << keypointDim_ << ")";
        std::cout << "\n";

        embeddedClassNames_ = onnxmeta::tryGetEmbeddedClassNames(session_);
    }
};

} // namespace rfdetrs
