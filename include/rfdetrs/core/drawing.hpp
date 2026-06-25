// SPDX-FileCopyrightText: 2024 Geekgineer
// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

// ============================================================================
// RFDETRs-CPP Drawing Utilities
// ============================================================================

#include <opencv2/opencv.hpp>
#include <string>
#include <cmath>

#include "rfdetrs/core/types.hpp"

namespace rfdetrs {
namespace drawing {

inline void drawBoundingBox(cv::Mat& image,
                            const BoundingBox& box,
                            const std::string& label,
                            const cv::Scalar& color,
                            int thickness = 2) {
    cv::rectangle(image,
                  cv::Point(box.x, box.y),
                  cv::Point(box.x + box.width, box.y + box.height),
                  color, thickness, cv::LINE_AA);

    if (!label.empty()) {
        const int fontFace = cv::FONT_HERSHEY_SIMPLEX;
        const double fontScale = std::max(0.4, std::min(image.rows, image.cols) * 0.0008);
        const int textThick = std::max(1, static_cast<int>(std::min(image.rows, image.cols) * 0.002));
        int baseline = 0;
        cv::Size textSize = cv::getTextSize(label, fontFace, fontScale, textThick, &baseline);

        int labelY = std::max(box.y, textSize.height + 5);
        cv::rectangle(image,
                      cv::Point(box.x, labelY - textSize.height - 5),
                      cv::Point(box.x + textSize.width + 5, labelY + baseline - 5),
                      color, cv::FILLED);
        cv::putText(image, label, cv::Point(box.x + 2, labelY - 2),
                    fontFace, fontScale, cv::Scalar(255, 255, 255), textThick, cv::LINE_AA);
    }
}

inline void drawBoundingBoxWithMask(cv::Mat& image,
                                    const BoundingBox& box,
                                    const std::string& label,
                                    const cv::Scalar& color,
                                    float alpha = 0.4f) {
    cv::Mat overlay = image.clone();
    cv::rectangle(overlay, cv::Rect(box.x, box.y, box.width, box.height), color, cv::FILLED);
    cv::addWeighted(overlay, alpha, image, 1.0f - alpha, 0, image);
    drawBoundingBox(image, box, label, color, 2);
}

} // namespace drawing
} // namespace rfdetrs
