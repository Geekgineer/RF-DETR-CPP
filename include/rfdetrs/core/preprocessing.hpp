// SPDX-FileCopyrightText: 2024 Geekgineer
// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

// ============================================================================
// RFDETRs-CPP Preprocessing
// ============================================================================
// RF-DETR uses a plain stretch resize (NOT letterbox) followed by ImageNet
// mean/std normalization.  This is confirmed from rfdetr/src/rfdetr/detr.py:
//   batch_tensor = torch.stack([F.resize(t, resize_to) for t in ...])
//   batch_tensor = F.normalize(batch_tensor, self.means, self.stds)
//
// mean = [0.485, 0.456, 0.406]  std = [0.229, 0.224, 0.225]  (RGB order)
// ============================================================================

#include <opencv2/opencv.hpp>
#include <vector>
#include <cstddef>

namespace rfdetrs {
namespace preprocessing {

// ImageNet normalization constants (RGB order)
static constexpr float IMAGENET_MEAN[3] = {0.485f, 0.456f, 0.406f};
static constexpr float IMAGENET_STD[3]  = {0.229f, 0.224f, 0.225f};

// Pre-computed scale factors:  1/(255*std[c])  and  -mean[c]/std[c]
// normalized = (pixel/255 - mean) / std  =  pixel * (1/(255*std)) + (-mean/std)
static constexpr float INV_SCALE[3] = {
    1.0f / (255.0f * IMAGENET_STD[0]),   // R
    1.0f / (255.0f * IMAGENET_STD[1]),   // G
    1.0f / (255.0f * IMAGENET_STD[2]),   // B
};
static constexpr float BIAS[3] = {
    -IMAGENET_MEAN[0] / IMAGENET_STD[0], // R
    -IMAGENET_MEAN[1] / IMAGENET_STD[1], // G
    -IMAGENET_MEAN[2] / IMAGENET_STD[2], // B
};

// Portable restrict qualifier for auto-vectorization hints.
#if defined(__GNUC__) || defined(__clang__)
#  define RFDETRS_RESTRICT __restrict__
#elif defined(_MSC_VER)
#  define RFDETRS_RESTRICT __restrict
#else
#  define RFDETRS_RESTRICT
#endif

/// @brief Stretch-resize a BGR image to targetSize, convert BGR→RGB, normalize
///        with ImageNet mean/std, and write to a pre-allocated CHW float blob.
///
/// @param bgrImage  Input image in OpenCV BGR format.
/// @param blob      Output blob, resized to 3*H*W. CHW layout, channel order R,G,B.
/// @param targetSize  Target (W, H) — the model's static input resolution.
inline void resizeAndNormalize(const cv::Mat& bgrImage,
                               std::vector<float>& blob,
                               const cv::Size& targetSize) {
    const int W = targetSize.width;
    const int H = targetSize.height;
    const int N = H * W;
    const size_t total = static_cast<size_t>(3 * N);

    if (blob.size() != total) blob.resize(total);

    // Stretch resize (bilinear), matches torchvision.transforms.functional.resize
    cv::Mat resized;
    if (bgrImage.cols != W || bgrImage.rows != H)
        cv::resize(bgrImage, resized, targetSize, 0, 0, cv::INTER_LINEAR);
    else
        resized = bgrImage;

    // cv::split de-interleaves BGR into three contiguous planes using OpenCV's
    // SIMD backend. The resulting per-plane loops are pure linear scans with no
    // stride/gather, which the compiler can auto-vectorize (AVX2/NEON) with
    // -O3 -march=native far better than the original interleaved BGR gather.
    cv::Mat planes[3];
    cv::split(resized, planes);  // planes[0]=B, planes[1]=G, planes[2]=R

    float*        RFDETRS_RESTRICT dstR = blob.data();
    float*        RFDETRS_RESTRICT dstG = blob.data() + N;
    float*        RFDETRS_RESTRICT dstB = blob.data() + 2 * N;
    const uchar*  RFDETRS_RESTRICT srcR = planes[2].data;
    const uchar*  RFDETRS_RESTRICT srcG = planes[1].data;
    const uchar*  RFDETRS_RESTRICT srcB = planes[0].data;

    for (int i = 0; i < N; ++i) dstR[i] = static_cast<float>(srcR[i]) * INV_SCALE[0] + BIAS[0];
    for (int i = 0; i < N; ++i) dstG[i] = static_cast<float>(srcG[i]) * INV_SCALE[1] + BIAS[1];
    for (int i = 0; i < N; ++i) dstB[i] = static_cast<float>(srcB[i]) * INV_SCALE[2] + BIAS[2];
}

} // namespace preprocessing
} // namespace rfdetrs
