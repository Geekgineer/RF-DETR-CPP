// SPDX-FileCopyrightText: 2024 Geekgineer
// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

// ============================================================================
// RFDETRs-CPP — RF-DETR Segmentation
// ============================================================================
// ONNX export contract (rf-detr ≥ 1.6, segmentation variants):
//   Input  "input"  : float32[1, 3, H, W]
//   Output "dets"   : float32[1, N, 4]        (cxcywh normalized)
//   Output "labels" : float32[1, N, C]        (class logits)
//   Output "masks"  : float32[1, N, H/r, W/r] (mask logits, r=mask_downsample_ratio=4)
//
// Postprocess (mirrors postprocess.py _postprocess_masks):
//   1. Same topK selection over flattened sigmoid(labels) as detection.
//   2. For each selected (query_idx, class_idx, score):
//        mask_logits = masks[0, query_idx, :, :]        — [H/r, W/r]
//        mask_resized = bilinear_upsample(mask_logits, (origH, origW))
//        mask_bool = mask_resized > 0.0f                — threshold in logit space
//   3. Return SegmentationDetection{box, conf, classId, mask (CV_8UC1)}.
// ============================================================================

#include <opencv2/imgproc.hpp>
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
namespace seg {

class RFDETRSegDetector : public OrtSessionBase {
public:
    struct TimingMs {
        double preprocess{0.0};
        double inference{0.0};
        double postprocess{0.0};
        double total() const noexcept { return preprocess + inference + postprocess; }
    };

    /// @param modelPath   Path to the segmentation ONNX model (e.g. rfdetr-seg-nano.onnx)
    /// @param labelsPath  Path to .names file (empty → class IDs only)
    /// @param backend     CPU (default), CUDA, or TensorRT
    /// @param numThreads  Intra-op threads (0 = auto)
    RFDETRSegDetector(const std::string& modelPath,
                      const std::string& labelsPath = "",
                      BackendType backend = BackendType::CPU,
                      int numThreads = 0)
        : OrtSessionBase(modelPath, backend, numThreads) {
        if (maskOutputIdx_ < 0)
            throw std::runtime_error(
                "Model has no 'masks' output — load a segmentation model "
                "(e.g. rfdetr-seg-nano.onnx), not a detection model.");

        classNames_ = utils::getClassNames(labelsPath);
        classColors_ = utils::generateColors(
            classNames_.empty()
                ? std::vector<std::string>(static_cast<size_t>(numClasses_), "")
                : classNames_);
    }

    /// Run segmentation inference on a BGR image.
    std::vector<SegmentationDetection> detect(const cv::Mat& image,
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
                                   outputs[maskOutputIdx_],
                                   confThreshold, topK);

        auto t3 = HrClock::now();

        lastTiming_.preprocess  = Ms(t1 - t0).count();
        lastTiming_.inference   = Ms(t2 - t1).count();
        lastTiming_.postprocess = Ms(t3 - t2).count();

        return results;
    }

    [[nodiscard]] const TimingMs& getLastTiming() const noexcept { return lastTiming_; }

    void drawDetections(cv::Mat& image,
                        const std::vector<SegmentationDetection>& dets,
                        float maskAlpha = 0.4f) const {
        for (const auto& d : dets) {
            if (d.classId < 0) continue;
            const cv::Scalar& color = classColors_[d.classId % classColors_.size()];

            // Overlay mask: blend colour only where mask is active.
            // addWeighted on the full image then copyTo(mask) keeps pixels
            // outside the mask untouched.
            if (!d.mask.empty() && d.mask.size() == image.size()) {
                cv::Mat colored(image.size(), CV_8UC3, color);
                cv::Mat blended;
                cv::addWeighted(image, 1.0 - maskAlpha, colored, maskAlpha, 0, blended);
                blended.copyTo(image, d.mask);
            }

            std::string label = (d.classId < static_cast<int>(classNames_.size()))
                ? classNames_[d.classId] + ": " + std::to_string(static_cast<int>(d.conf * 100)) + "%"
                : "cls" + std::to_string(d.classId) + ": " + std::to_string(static_cast<int>(d.conf * 100)) + "%";
            drawing::drawBoundingBox(image, d.box, label, color);
        }
    }

    [[nodiscard]] const std::vector<std::string>& getClassNames() const { return classNames_; }
    [[nodiscard]] const std::vector<cv::Scalar>&  getClassColors() const { return classColors_; }

private:
    std::vector<std::string> classNames_;
    std::vector<cv::Scalar>  classColors_;
    mutable TimingMs         lastTiming_;

    std::vector<SegmentationDetection> postprocess(
            int origW, int origH,
            const Ort::Value& boxTensor,
            const Ort::Value& logitTensor,
            const Ort::Value& maskTensor,
            float confThreshold, int topK) const {

        const float* boxes  = boxTensor.GetTensorData<float>();
        const float* logits = logitTensor.GetTensorData<float>();
        const float* masks  = maskTensor.GetTensorData<float>();

        const int N        = numQueries_;
        const int C        = numClasses_;
        const int flatSize = N * C;
        const int Hm       = maskH_;
        const int Wm       = maskW_;

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

        std::vector<SegmentationDetection> results;
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

            // Gather mask for this query: masks[0, qi, :, :] — shape [Hm, Wm]
            // Memory layout: [1, N, Hm, Wm] → offset = qi * Hm * Wm
            const float* maskPtr = masks + qi * Hm * Wm;

            // Wrap raw mask logits in cv::Mat (no copy) and upsample to orig size
            cv::Mat maskLogits(Hm, Wm, CV_32FC1, const_cast<float*>(maskPtr));
            cv::Mat maskResized;
            cv::resize(maskLogits, maskResized, cv::Size(origW, origH), 0, 0, cv::INTER_LINEAR);

            // Threshold at 0.0 in logit space (equivalent to sigmoid > 0.5)
            cv::Mat maskBool;
            cv::threshold(maskResized, maskBool, 0.0, 255.0, cv::THRESH_BINARY);
            maskBool.convertTo(maskBool, CV_8UC1);

            results.emplace_back(BoundingBox{ix1, iy1, ix2 - ix1, iy2 - iy1},
                                 score, ci, std::move(maskBool));
        }

        return results;
    }
};

} // namespace seg
} // namespace rfdetrs
