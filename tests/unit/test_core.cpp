// SPDX-FileCopyrightText: 2024 Geekgineer
// SPDX-License-Identifier: AGPL-3.0-only
//
// ============================================================================
// RFDETRs-CPP — Unit tests for core math helpers and preprocessing
// ============================================================================
// These tests require no ONNX model and no GPU — they exercise the
// header-only hot-path code that is most likely to regress across
// optimisation passes.
// ============================================================================

#include <gtest/gtest.h>
#include <opencv2/imgproc.hpp>

#include "rfdetrs/core/preprocessing.hpp"
#include "rfdetrs/core/utils.hpp"

using namespace rfdetrs;

// ============================================================================
// utils::clamp
// ============================================================================
TEST(ClampTest, BelowRange)  { EXPECT_EQ(utils::clamp( 0, 5, 10), 5);  }
TEST(ClampTest, AboveRange)  { EXPECT_EQ(utils::clamp(15, 5, 10), 10); }
TEST(ClampTest, InRange)     { EXPECT_EQ(utils::clamp( 7, 5, 10), 7);  }
TEST(ClampTest, AtLowBound)  { EXPECT_EQ(utils::clamp( 5, 5, 10), 5);  }
TEST(ClampTest, AtHighBound) { EXPECT_EQ(utils::clamp(10, 5, 10), 10); }

TEST(ClampTest, FloatInRange) {
    EXPECT_NEAR(utils::clamp(0.7f, 0.0f, 1.0f), 0.7f, 1e-6f);
}
TEST(ClampTest, FloatBelow) {
    EXPECT_NEAR(utils::clamp(-0.1f, 0.0f, 1.0f), 0.0f, 1e-6f);
}
TEST(ClampTest, FloatAbove) {
    EXPECT_NEAR(utils::clamp(1.5f, 0.0f, 1.0f), 1.0f, 1e-6f);
}
TEST(ClampTest, NegativeRange) {
    EXPECT_EQ(utils::clamp(-3, -5, -1), -3);
    EXPECT_EQ(utils::clamp(-9, -5, -1), -5);
    EXPECT_EQ(utils::clamp( 0, -5, -1), -1);
}

// ============================================================================
// utils::sigmoid
// ============================================================================
TEST(SigmoidTest, AtZeroIsHalf) {
    EXPECT_NEAR(utils::sigmoid(0.0f), 0.5f, 1e-6f);
}
TEST(SigmoidTest, LargePositiveNearOne) {
    EXPECT_GT(utils::sigmoid(50.0f), 0.9999f);
}
TEST(SigmoidTest, LargeNegativeNearZero) {
    EXPECT_LT(utils::sigmoid(-50.0f), 0.0001f);
}
TEST(SigmoidTest, OutputInUnitInterval) {
    for (float x : {-10.f, -1.f, 0.f, 1.f, 10.f}) {
        float s = utils::sigmoid(x);
        EXPECT_GT(s, 0.0f) << "x=" << x;
        EXPECT_LT(s, 1.0f) << "x=" << x;
    }
}
TEST(SigmoidTest, Monotone) {
    EXPECT_LT(utils::sigmoid(-2.0f), utils::sigmoid(-1.0f));
    EXPECT_LT(utils::sigmoid(-1.0f), utils::sigmoid( 0.0f));
    EXPECT_LT(utils::sigmoid( 0.0f), utils::sigmoid( 1.0f));
    EXPECT_LT(utils::sigmoid( 1.0f), utils::sigmoid( 2.0f));
}
TEST(SigmoidTest, Antisymmetry) {
    // sigmoid(-x) == 1 - sigmoid(x)
    for (float x : {0.5f, 1.0f, 2.0f, 5.0f}) {
        EXPECT_NEAR(utils::sigmoid(-x), 1.0f - utils::sigmoid(x), 1e-6f)
            << "x=" << x;
    }
}
TEST(SigmoidTest, KnownValues) {
    // sigmoid(1)  ≈ 0.7311
    EXPECT_NEAR(utils::sigmoid(1.0f), 0.731059f, 1e-5f);
    // sigmoid(-1) ≈ 0.2689
    EXPECT_NEAR(utils::sigmoid(-1.0f), 0.268941f, 1e-5f);
}

// ============================================================================
// utils::logit  (inverse of sigmoid)
// ============================================================================
TEST(LogitTest, AtHalfIsZero) {
    EXPECT_NEAR(utils::logit(0.5f), 0.0f, 1e-6f);
}
TEST(LogitTest, InverseOfSigmoid) {
    for (float p : {0.05f, 0.1f, 0.3f, 0.5f, 0.7f, 0.9f, 0.95f}) {
        EXPECT_NEAR(utils::sigmoid(utils::logit(p)), p, 1e-5f)
            << "round-trip failed for p=" << p;
    }
}
TEST(LogitTest, SigmoidOfLogitIsIdentity) {
    for (float x : {-3.f, -1.f, 0.f, 1.f, 3.f}) {
        EXPECT_NEAR(utils::logit(utils::sigmoid(x)), x, 1e-4f)
            << "round-trip failed for x=" << x;
    }
}
TEST(LogitTest, Monotone) {
    EXPECT_LT(utils::logit(0.1f), utils::logit(0.5f));
    EXPECT_LT(utils::logit(0.5f), utils::logit(0.9f));
}
TEST(LogitTest, ThresholdMapping) {
    // conf=0.5 → logit=0; everything below 0.5 has negative logit
    EXPECT_NEAR(utils::logit(0.5f), 0.0f, 1e-6f);
    EXPECT_LT(utils::logit(0.3f), 0.0f);
    EXPECT_GT(utils::logit(0.7f), 0.0f);
}

// ============================================================================
// utils::trimString
// ============================================================================
TEST(TrimStringTest, NoSpaces)      { EXPECT_EQ(utils::trimString("hello"),    "hello"); }
TEST(TrimStringTest, LeadingSpace)  { EXPECT_EQ(utils::trimString("  hello"),  "hello"); }
TEST(TrimStringTest, TrailingSpace) { EXPECT_EQ(utils::trimString("hello  "),  "hello"); }
TEST(TrimStringTest, BothEnds)      { EXPECT_EQ(utils::trimString("  hello  "),"hello"); }
TEST(TrimStringTest, EmptyString)   { EXPECT_EQ(utils::trimString(""),         "");      }
TEST(TrimStringTest, AllSpaces)     { EXPECT_EQ(utils::trimString("   "),      "");      }
TEST(TrimStringTest, TabNewline) {
    EXPECT_EQ(utils::trimString("\t hello \n"), "hello");
}

// ============================================================================
// preprocessing::resizeAndNormalize
// ============================================================================

// Helper: expected normalized value for a uint8 pixel level p, channel c.
static float expectedNorm(int p, int c) {
    using namespace rfdetrs::preprocessing;
    return static_cast<float>(p) * INV_SCALE[c] + BIAS[c];
}

TEST(PreprocessTest, OutputBlobSize) {
    cv::Mat img = cv::Mat::zeros(100, 200, CV_8UC3);
    std::vector<float> blob;
    preprocessing::resizeAndNormalize(img, blob, cv::Size(560, 560));
    EXPECT_EQ(blob.size(), static_cast<size_t>(3 * 560 * 560));
}

TEST(PreprocessTest, BlobResizedWhenWrongCapacity) {
    cv::Mat img = cv::Mat::zeros(10, 10, CV_8UC3);
    std::vector<float> blob(100, 0.0f);  // wrong initial size
    preprocessing::resizeAndNormalize(img, blob, cv::Size(20, 30));
    EXPECT_EQ(blob.size(), static_cast<size_t>(3 * 20 * 30));
}

TEST(PreprocessTest, BlackPixelValues) {
    // BGR (0,0,0) → all channels zero before normalization
    cv::Mat img(1, 1, CV_8UC3, cv::Scalar(0, 0, 0));
    std::vector<float> blob;
    preprocessing::resizeAndNormalize(img, blob, cv::Size(1, 1));
    ASSERT_EQ(blob.size(), 3u);
    EXPECT_NEAR(blob[0], expectedNorm(0, 0), 1e-4f);  // R plane
    EXPECT_NEAR(blob[1], expectedNorm(0, 1), 1e-4f);  // G plane
    EXPECT_NEAR(blob[2], expectedNorm(0, 2), 1e-4f);  // B plane
}

TEST(PreprocessTest, WhitePixelValues) {
    // BGR (255,255,255)
    cv::Mat img(1, 1, CV_8UC3, cv::Scalar(255, 255, 255));
    std::vector<float> blob;
    preprocessing::resizeAndNormalize(img, blob, cv::Size(1, 1));
    ASSERT_EQ(blob.size(), 3u);
    EXPECT_NEAR(blob[0], expectedNorm(255, 0), 1e-4f);
    EXPECT_NEAR(blob[1], expectedNorm(255, 1), 1e-4f);
    EXPECT_NEAR(blob[2], expectedNorm(255, 2), 1e-4f);
}

TEST(PreprocessTest, CHWLayoutAndBGRToRGB) {
    // Pure-red pixel in BGR: B=0, G=0, R=255  →  cv::Scalar(0, 0, 255)
    cv::Mat img(1, 1, CV_8UC3, cv::Scalar(0, 0, 255));
    std::vector<float> blob;
    preprocessing::resizeAndNormalize(img, blob, cv::Size(1, 1));
    ASSERT_EQ(blob.size(), 3u);
    // CHW layout: blob[0]=R plane, blob[1]=G plane, blob[2]=B plane
    EXPECT_NEAR(blob[0], expectedNorm(255, 0), 1e-4f) << "R plane wrong";
    EXPECT_NEAR(blob[1], expectedNorm(0,   1), 1e-4f) << "G plane wrong";
    EXPECT_NEAR(blob[2], expectedNorm(0,   2), 1e-4f) << "B plane wrong";
}

TEST(PreprocessTest, PureBluePureGreen) {
    // Pure blue:  BGR (255, 0, 0)
    cv::Mat blue(1, 1, CV_8UC3, cv::Scalar(255, 0, 0));
    std::vector<float> blob;
    preprocessing::resizeAndNormalize(blue, blob, cv::Size(1, 1));
    EXPECT_NEAR(blob[0], expectedNorm(0,   0), 1e-4f) << "blue: R plane";
    EXPECT_NEAR(blob[1], expectedNorm(0,   1), 1e-4f) << "blue: G plane";
    EXPECT_NEAR(blob[2], expectedNorm(255, 2), 1e-4f) << "blue: B plane";

    // Pure green: BGR (0, 255, 0)
    cv::Mat green(1, 1, CV_8UC3, cv::Scalar(0, 255, 0));
    preprocessing::resizeAndNormalize(green, blob, cv::Size(1, 1));
    EXPECT_NEAR(blob[0], expectedNorm(0,   0), 1e-4f) << "green: R plane";
    EXPECT_NEAR(blob[1], expectedNorm(255, 1), 1e-4f) << "green: G plane";
    EXPECT_NEAR(blob[2], expectedNorm(0,   2), 1e-4f) << "green: B plane";
}

TEST(PreprocessTest, MultiPixelCHWOrder) {
    // 1×4 image, all same colour: checks plane offsets
    const uchar B = 50, G = 100, R = 200;
    cv::Mat img(1, 4, CV_8UC3, cv::Scalar(B, G, R));
    std::vector<float> blob;
    preprocessing::resizeAndNormalize(img, blob, cv::Size(4, 1));
    // Plane 0 (R): blob[0..3], Plane 1 (G): blob[4..7], Plane 2 (B): blob[8..11]
    for (int i = 0; i < 4; ++i) {
        EXPECT_NEAR(blob[i + 0], expectedNorm(R, 0), 1e-4f) << "R pixel " << i;
        EXPECT_NEAR(blob[i + 4], expectedNorm(G, 1), 1e-4f) << "G pixel " << i;
        EXPECT_NEAR(blob[i + 8], expectedNorm(B, 2), 1e-4f) << "B pixel " << i;
    }
}

TEST(PreprocessTest, SkipsResizeWhenSizeMatches) {
    // When input already matches target, the output should reflect exact values
    const uchar B = 10, G = 20, R = 30;
    cv::Mat img(3, 3, CV_8UC3, cv::Scalar(B, G, R));
    std::vector<float> blob;
    preprocessing::resizeAndNormalize(img, blob, cv::Size(3, 3));
    EXPECT_EQ(blob.size(), static_cast<size_t>(3 * 9));
    EXPECT_NEAR(blob[0], expectedNorm(R, 0), 1e-4f);
}
