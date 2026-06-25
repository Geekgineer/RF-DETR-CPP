// SPDX-FileCopyrightText: 2024 Geekgineer
// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

// ============================================================================
// RFDETRs-CPP — RF-DETR Keypoint Detection
// ============================================================================
// ONNX export contract (rf-detr ≥ 1.8, keypoint variants):
//   Input  "input"     : float32[1, 3, H, W]
//   Output "dets"      : float32[1, N, 4]            (cxcywh normalized)
//   Output "labels"    : float32[1, N, C]            (class logits)
//   Output "keypoints" : float32[1, N, C*max_K, 8]  (raw GroupPose keypoints)
//
// KEYPOINT_PRED_DIM = 8 layout per slot (GroupPose NLL-Cholesky head):
//   [0] x_norm            — keypoint x in [0,1]
//   [1] y_norm            — keypoint y in [0,1]
//   [2] findable_logit    — sigmoid → findable confidence
//   [3] visible_logit     — sigmoid → visible confidence
//   [4] L_xx, [5] L_xy, [6] L_yy — Cholesky precision (log-diagonal)
//   [7] class_logit       — class-specific logit (unused in basic decode)
//
// Slot layout — class-major (matches Python reshape to [K, C, max_K, D]):
//   Slots [c * max_K, (c+1) * max_K) belong to class c.
//   Only the first num_keypoints_per_class[c] slots are meaningful.
//
// Postprocess (mirrors postprocess.py _postprocess_keypoints):
//   1. Same topK selection as detection.
//   2. For each selected (query_idx, class_idx, score):
//        kps = keypoints[0, query_idx, :, :]   — [C*max_K, 8]
//        offset = class_idx * max_K             — class-major row offset
//        For k in [0, num_keypoints_per_class[class_idx]):
//          x = kps[offset + k, 0] * origW
//          y = kps[offset + k, 1] * origH
//          conf = sigmoid(kps[offset + k, 2])
//          vis  = sigmoid(kps[offset + k, 3])
//
// Note: The COCO KeypointPreview model uses
//   num_keypoints_per_class = {0, 17}   (class 0 = background, class 1 = person)
// Pass this via the constructor.  If empty, all classes are assumed to have
// max(keypointSlots / numClasses) keypoints (uniform fallback).
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
namespace kp {

class RFDETRKeypointDetector : public OrtSessionBase {
public:
    struct TimingMs {
        double preprocess{0.0};
        double inference{0.0};
        double postprocess{0.0};
        double total() const noexcept { return preprocess + inference + postprocess; }
    };

    /// @param modelPath              Path to the keypoint ONNX model
    /// @param labelsPath             Path to .names file (empty → class IDs only)
    /// @param numKeypointsPerClass   Per-class keypoint counts (e.g. {0, 17} for COCO).
    ///                               Length must equal numClasses.  Pass {} to use uniform fallback.
    /// @param backend                CPU (default), CUDA, or TensorRT
    /// @param numThreads             Intra-op threads (0 = auto)
    RFDETRKeypointDetector(const std::string& modelPath,
                           const std::string& labelsPath = "",
                           std::vector<int> numKeypointsPerClass = {},
                           BackendType backend = BackendType::CPU,
                           int numThreads = 0)
        : OrtSessionBase(modelPath, backend, numThreads),
          numKpsPerClass_(std::move(numKeypointsPerClass)) {
        if (keypointOutputIdx_ < 0)
            throw std::runtime_error(
                "Model has no 'keypoints' output — load a keypoint model "
                "(e.g. rfdetr-keypoint-preview.onnx), not a detection model.");

        // Validate or build per-class keypoint counts.
        // The schema may be shorter than numClasses (classes beyond the schema have 0 kps).
        if (numKpsPerClass_.empty()) {
            // Uniform fallback: every class gets keypointSlots / numClasses slots.
            // Matches the padded layout [C, max_K, D] regardless of which classes are active.
            int maxK = (numClasses_ > 0) ? keypointSlots_ / numClasses_ : 0;
            numKpsPerClass_.assign(static_cast<size_t>(numClasses_), maxK);
        } else if (static_cast<int>(numKpsPerClass_.size()) > numClasses_) {
            throw std::runtime_error(
                "numKeypointsPerClass length (" +
                std::to_string(numKpsPerClass_.size()) +
                ") exceeds numClasses (" + std::to_string(numClasses_) + ").");
        }

        // Precompute max keypoints and class slot offsets.
        // The keypoints tensor layout is class-major: [K, C_schema * max_K, D]
        // where C_schema = keypointSlots_ / max_K.
        // Class c occupies row c of the C_schema × max_K grid → offset = c * max_K.
        maxKp_ = 0;
        for (int n : numKpsPerClass_) maxKp_ = std::max(maxKp_, n);
        if (maxKp_ == 0 && keypointSlots_ > 0 && numClasses_ > 0)
            maxKp_ = keypointSlots_ / numClasses_;  // fallback from tensor shape

        const int schemaClasses = static_cast<int>(numKpsPerClass_.size());
        classKpOffset_.resize(static_cast<size_t>(schemaClasses), 0);
        for (int c = 0; c < schemaClasses; ++c)
            classKpOffset_[static_cast<size_t>(c)] = c * maxKp_;

        classNames_ = utils::getClassNames(labelsPath);
        classColors_ = utils::generateColors(
            classNames_.empty()
                ? std::vector<std::string>(static_cast<size_t>(numClasses_), "")
                : classNames_);
    }

    /// Run keypoint detection on a BGR image.
    std::vector<KeypointDetection> detect(const cv::Mat& image,
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
                                   outputs[boxOutputIdx_],
                                   outputs[logitOutputIdx_],
                                   outputs[keypointOutputIdx_],
                                   confThreshold, topK);

        auto t3 = HrClock::now();

        lastTiming_.preprocess  = Ms(t1 - t0).count();
        lastTiming_.inference   = Ms(t2 - t1).count();
        lastTiming_.postprocess = Ms(t3 - t2).count();

        return results;
    }

    [[nodiscard]] const TimingMs& getLastTiming() const noexcept { return lastTiming_; }

    void drawDetections(cv::Mat& image,
                        const std::vector<KeypointDetection>& dets,
                        float confCutoff = 0.3f) const {
        for (const auto& d : dets) {
            if (d.classId < 0) continue;
            const cv::Scalar& color = classColors_[d.classId % classColors_.size()];
            std::string label = (d.classId < static_cast<int>(classNames_.size()))
                ? classNames_[d.classId] + ": " + std::to_string(static_cast<int>(d.conf * 100)) + "%"
                : "cls" + std::to_string(d.classId) + ": " + std::to_string(static_cast<int>(d.conf * 100)) + "%";
            drawing::drawBoundingBox(image, d.box, label, color);

            for (const auto& kp : d.keypoints) {
                if (kp.confidence < confCutoff) continue;
                cv::circle(image, cv::Point(static_cast<int>(kp.x), static_cast<int>(kp.y)),
                           4, color, -1);
            }
        }
    }

    [[nodiscard]] const std::vector<std::string>& getClassNames()    const { return classNames_; }
    [[nodiscard]] const std::vector<cv::Scalar>&  getClassColors()   const { return classColors_; }
    [[nodiscard]] const std::vector<int>& getNumKeypointsPerClass()  const { return numKpsPerClass_; }
    [[nodiscard]] int getMaxKeypoints()                               const { return maxKp_; }

private:
    std::vector<std::string> classNames_;
    std::vector<cv::Scalar>  classColors_;
    std::vector<int>         numKpsPerClass_;
    std::vector<int>         classKpOffset_;  ///< Cumulative slot offset per class
    int                      maxKp_{0};       ///< Max keypoints any class has
    mutable TimingMs         lastTiming_;

    std::vector<KeypointDetection> postprocess(
            int origW, int origH,
            const Ort::Value& boxTensor,
            const Ort::Value& logitTensor,
            const Ort::Value& keypointTensor,
            float confThreshold, int topK) const {

        const float* boxes     = boxTensor.GetTensorData<float>();
        const float* logits    = logitTensor.GetTensorData<float>();
        const float* kpData    = keypointTensor.GetTensorData<float>();

        const int N        = numQueries_;
        const int C        = numClasses_;
        const int S        = keypointSlots_;  // C * max_K slots per query
        const int D        = keypointDim_;    // KEYPOINT_PRED_DIM (8)
        const int flatSize = N * C;

        const float logitThresh = (confThreshold <= 0.f) ? -1e9f
                                : (confThreshold >= 1.f) ?  1e9f
                                : utils::logit(confThreshold);

        using ScoreIdx = std::pair<float, int>;
        static thread_local std::vector<ScoreIdx> scored;
        scored.clear();

        for (int i = 0; i < flatSize; ++i) {
            if (logits[i] >= logitThresh)
                scored.emplace_back(logits[i], i);
        }

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

        std::vector<KeypointDetection> results;
        results.reserve(scored.size());

        for (const auto& si : scored) {
            const float score = utils::sigmoid(si.first);
            const int flatIdx = si.second;
            const int qi      = flatIdx / C;
            const int ci      = flatIdx % C;

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

            // Gather keypoints for this query: kpData[qi, :, :] — shape [S, D]
            // Memory layout: [1, N, S, D] → offset = qi * S * D
            const float* qKp = kpData + qi * S * D;

            // Extract class-specific keypoints using precomputed slot offset.
            int numKp  = (ci < static_cast<int>(numKpsPerClass_.size()))
                       ? numKpsPerClass_[static_cast<size_t>(ci)] : 0;
            int offset = (ci < static_cast<int>(classKpOffset_.size()))
                       ? classKpOffset_[static_cast<size_t>(ci)] : 0;

            std::vector<Keypoint> kps;
            kps.reserve(static_cast<size_t>(numKp));
            for (int k2 = 0; k2 < numKp; ++k2) {
                const float* slot = qKp + (offset + k2) * D;
                Keypoint kp;
                kp.x          = slot[0] * static_cast<float>(origW);
                kp.y          = slot[1] * static_cast<float>(origH);
                kp.confidence = utils::sigmoid(slot[2]);
                kp.visible    = utils::sigmoid(slot[3]);
                kps.push_back(kp);
            }

            results.emplace_back(BoundingBox{ix1, iy1, ix2 - ix1, iy2 - iy1},
                                 score, ci, std::move(kps));
        }

        return results;
    }
};

} // namespace kp
} // namespace rfdetrs
