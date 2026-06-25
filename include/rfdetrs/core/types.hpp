// SPDX-FileCopyrightText: 2024 Geekgineer
// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

// ============================================================================
// RFDETRs-CPP Core Types
// ============================================================================
// Shared data structures for RF-DETR detection results.
// ============================================================================

#include <algorithm>
#include <vector>
#include <opencv2/core.hpp>

namespace rfdetrs {

struct BoundingBox {
    int x{0};
    int y{0};
    int width{0};
    int height{0};

    BoundingBox() = default;
    BoundingBox(int x_, int y_, int width_, int height_)
        : x(x_), y(y_), width(width_), height(height_) {}

    [[nodiscard]] float area() const noexcept {
        return static_cast<float>(width * height);
    }

    [[nodiscard]] BoundingBox intersect(const BoundingBox& o) const noexcept {
        int x0 = std::max(x, o.x);
        int y0 = std::max(y, o.y);
        int x1 = std::min(x + width, o.x + o.width);
        int y1 = std::min(y + height, o.y + o.height);
        int iw = std::max(0, x1 - x0);
        int ih = std::max(0, y1 - y0);
        return {x0, y0, iw, ih};
    }

    [[nodiscard]] float iou(const BoundingBox& o) const noexcept {
        BoundingBox inter = intersect(o);
        float interArea = inter.area();
        float unionArea = area() + o.area() - interArea;
        return (unionArea > 0.0f) ? (interArea / unionArea) : 0.0f;
    }
};

namespace det {

struct Detection {
    BoundingBox box;
    float conf{0.0f};
    int classId{-1};

    Detection() = default;
    Detection(const BoundingBox& b, float c, int id) : box(b), conf(c), classId(id) {}
};

} // namespace det

// ============================================================================
// Segmentation types
// ============================================================================

namespace seg {

struct SegmentationDetection {
    BoundingBox box;
    float       conf{0.0f};
    int         classId{-1};
    cv::Mat     mask;  ///< CV_8UC1 boolean mask, same size as original image

    SegmentationDetection() = default;
    SegmentationDetection(const BoundingBox& b, float c, int id, cv::Mat m)
        : box(b), conf(c), classId(id), mask(std::move(m)) {}
};

} // namespace seg

// ============================================================================
// Keypoint types
// ============================================================================

namespace kp {

struct Keypoint {
    float x{0.f};           ///< Pixel-space x coordinate
    float y{0.f};           ///< Pixel-space y coordinate
    float confidence{0.f};  ///< sigmoid(findable_logit)
    float visible{0.f};     ///< sigmoid(visible_logit)
};

struct KeypointDetection {
    BoundingBox          box;
    float                conf{0.0f};
    int                  classId{-1};
    std::vector<Keypoint> keypoints;

    KeypointDetection() = default;
    KeypointDetection(const BoundingBox& b, float c, int id, std::vector<Keypoint> kps)
        : box(b), conf(c), classId(id), keypoints(std::move(kps)) {}
};

} // namespace kp

} // namespace rfdetrs
