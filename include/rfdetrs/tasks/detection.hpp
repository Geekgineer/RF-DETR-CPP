// SPDX-FileCopyrightText: 2024 Geekgineer
// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

// ============================================================================
// RFDETRs-CPP — RF-DETR Detection
// ============================================================================
// Implements OrtSessionBase (CPU / CUDA / TensorRT via BackendType enum) and
// RFDETRDetector, which mirrors the API of YOLOs-CPP's YOLODetector.
//
// ONNX export contract (confirmed from rf-detr source):
//   Input  "input"  : float32[1, 3, H, W]  (static shape)
//   Output "dets"   : float32[1, N, 4]     (cxcywh normalized [0,1])
//   Output "labels" : float32[1, N, C]     (class logits, C>4)
//
// Postprocess matches rfdetr/src/rfdetr/models/postprocess.py exactly:
//   prob = sigmoid(labels)                        # [1, N, C]
//   flat = prob.view(N*C)                         # flatten
//   topk → (score, flat_idx)                      # top-K (query,class) pairs
//   query_idx = flat_idx / C
//   class_idx = flat_idx % C
//   xyxy = cxcywh_to_xyxy(dets[query_idx])
//   pixel = xyxy * [W_orig, H_orig, W_orig, H_orig]
// ============================================================================

#include <opencv2/opencv.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "rfdetrs/core/drawing.hpp"
#include "rfdetrs/core/preprocessing.hpp"
#include "rfdetrs/core/session_base.hpp"
#include "rfdetrs/core/types.hpp"
#include "rfdetrs/core/utils.hpp"

namespace rfdetrs {

// ============================================================================
// RFDETRDetector
// ============================================================================

namespace det {

class RFDETRDetector : public OrtSessionBase {
public:
    /// Phase timing from the most recent detect() call (milliseconds).
    struct TimingMs {
        double preprocess{0.0};
        double inference{0.0};
        double postprocess{0.0};
        double total() const noexcept { return preprocess + inference + postprocess; }
    };

    /// @param modelPath   Path to the exported inference_model.onnx
    /// @param labelsPath  Path to a .names / .txt class list (one name per line).
    ///                    If empty, class IDs are used without names.
    /// @param backend     CPU (default), CUDA, or TensorRT
    /// @param numThreads  Intra-op threads (0 = auto)
    RFDETRDetector(const std::string& modelPath,
                   const std::string& labelsPath = "",
                   BackendType backend = BackendType::CPU,
                   int numThreads = 0)
        : OrtSessionBase(modelPath, backend, numThreads) {
        classNames_ = utils::getClassNames(labelsPath);
        classColors_ = utils::generateColors(
            classNames_.empty()
                ? std::vector<std::string>(static_cast<size_t>(numClasses_), "")
                : classNames_);
    }

    /// @brief Run detection on a BGR image.
    /// @param image          Input image (OpenCV BGR).
    /// @param confThreshold  Minimum sigmoid score to keep a detection.
    /// @param topK           Maximum (query, class) pairs to evaluate.
    /// @return Detections sorted by confidence descending.
    std::vector<Detection> detect(const cv::Mat& image,
                                  float confThreshold = 0.5f,
                                  int topK = 300) {
        using HrClock = std::chrono::high_resolution_clock;
        using Ms      = std::chrono::duration<double, std::milli>;

        auto t0 = HrClock::now();

        preprocessing::resizeAndNormalize(image, blob_, inputShape_);
        std::vector<int64_t> shape = {1, 3,
                                      static_cast<int64_t>(inputShape_.height),
                                      static_cast<int64_t>(inputShape_.width)};
        Ort::Value inputTensor = createInputTensor(shape);

        auto t1 = HrClock::now();

        std::vector<Ort::Value> outputs = runInference(inputTensor);

        auto t2 = HrClock::now();

        auto results = postprocess(image.cols, image.rows,
                                   outputs[boxOutputIdx_], outputs[logitOutputIdx_],
                                   confThreshold, topK);

        auto t3 = HrClock::now();

        lastTiming_.preprocess  = Ms(t1 - t0).count();
        lastTiming_.inference   = Ms(t2 - t1).count();
        lastTiming_.postprocess = Ms(t3 - t2).count();

        return results;
    }

    [[nodiscard]] const TimingMs& getLastTiming() const noexcept { return lastTiming_; }

    void drawDetections(cv::Mat& image, const std::vector<Detection>& dets) const {
        for (const auto& d : dets) {
            if (d.classId < 0) continue;
            std::string label = (d.classId < static_cast<int>(classNames_.size()))
                ? classNames_[d.classId] + ": " + std::to_string(static_cast<int>(d.conf * 100)) + "%"
                : "cls" + std::to_string(d.classId) + ": " + std::to_string(static_cast<int>(d.conf * 100)) + "%";
            const cv::Scalar& color = classColors_[d.classId % classColors_.size()];
            drawing::drawBoundingBox(image, d.box, label, color);
        }
    }

    [[nodiscard]] const std::vector<std::string>& getClassNames() const { return classNames_; }
    [[nodiscard]] const std::vector<cv::Scalar>&  getClassColors() const { return classColors_; }

private:
    std::vector<std::string> classNames_;
    std::vector<cv::Scalar>  classColors_;
    mutable TimingMs         lastTiming_;

    // Postprocess matches rfdetr/src/rfdetr/models/postprocess.py exactly.
    //
    // Optimizations over the naive implementation:
    //   1. Logit-space pre-filter: sigmoid is strictly monotone, so
    //      sigmoid(x) >= t  ⟺  x >= logit(t).  We skip exp() on the ~99% of
    //      the 300×91 = 27,300 elements that fall below the threshold.
    //   2. thread_local buffer: avoids a 27,300-element heap allocation each call.
    //   3. nth_element (O(N)) + sort top-K (O(K log K)) instead of
    //      partial_sort (O(N log K)) — faster when few elements survive the filter.
    //   4. Exact sigmoid only for the surviving ≤topK candidates.
    std::vector<Detection> postprocess(int origW, int origH,
                                       const Ort::Value& boxTensor,
                                       const Ort::Value& logitTensor,
                                       float confThreshold, int topK) const {
        const float* boxes  = boxTensor.GetTensorData<float>();
        const float* logits = logitTensor.GetTensorData<float>();

        const int N        = numQueries_;
        const int C        = numClasses_;
        const int flatSize = N * C;

        // Convert confidence threshold to logit space once per call.
        const float logitThresh = (confThreshold <= 0.f) ? -1e9f
                                : (confThreshold >= 1.f) ?  1e9f
                                : utils::logit(confThreshold);

        // Thread-local scratch avoids malloc/free on every detect() call.
        // Capacity grows on demand and is retained across calls on the same thread.
        using ScoreIdx = std::pair<float, int>;
        static thread_local std::vector<ScoreIdx> scored;
        scored.clear();

        // Pre-filter in logit space — skip exp() for all below-threshold elements.
        // Matching: prob = out_logits.sigmoid() ; logits_for_topk = prob.view(-1)
        for (int i = 0; i < flatSize; ++i) {
            if (logits[i] >= logitThresh)
                scored.emplace_back(logits[i], i);
        }

        // Find top-K boundary with O(N) nth_element, then sort only the survivors.
        // Matching: topk_values, topk_indexes = torch.topk(logits_for_topk, num_select)
        int k = std::min(topK, static_cast<int>(scored.size()));
        if (k < static_cast<int>(scored.size())) {
            std::nth_element(scored.begin(), scored.begin() + k, scored.end(),
                             [](const ScoreIdx& a, const ScoreIdx& b) {
                                 return a.first > b.first;
                             });
            scored.resize(static_cast<size_t>(k));
        }
        std::sort(scored.begin(), scored.end(),
                  [](const ScoreIdx& a, const ScoreIdx& b) { return a.first > b.first; });

        // Decode: compute exact sigmoid only for the surviving ≤topK candidates.
        // Matching: topk_boxes = topk_indexes // C
        //           labels     = topk_indexes %  C
        //           boxes      = box_cxcywh_to_xyxy(out_bbox)[topk_boxes]
        //           boxes      *= [img_w, img_h, img_w, img_h]
        std::vector<Detection> results;
        results.reserve(scored.size());

        for (const auto& si : scored) {
            const float score   = utils::sigmoid(si.first);
            const int flatIdx   = si.second;
            const int qi        = flatIdx / C;
            const int ci        = flatIdx % C;

            const float cx = boxes[qi * 4 + 0];
            const float cy = boxes[qi * 4 + 1];
            const float bw = boxes[qi * 4 + 2];
            const float bh = boxes[qi * 4 + 3];

            const float x1 = (cx - bw * 0.5f) * static_cast<float>(origW);
            const float y1 = (cy - bh * 0.5f) * static_cast<float>(origH);
            const float x2 = (cx + bw * 0.5f) * static_cast<float>(origW);
            const float y2 = (cy + bh * 0.5f) * static_cast<float>(origH);

            const int ix1 = utils::clamp(static_cast<int>(x1), 0, origW - 1);
            const int iy1 = utils::clamp(static_cast<int>(y1), 0, origH - 1);
            const int ix2 = utils::clamp(static_cast<int>(x2), 0, origW);
            const int iy2 = utils::clamp(static_cast<int>(y2), 0, origH);

            results.emplace_back(
                BoundingBox{ix1, iy1, ix2 - ix1, iy2 - iy1},
                score, ci);
        }

        return results;
    }
};

} // namespace det
} // namespace rfdetrs
