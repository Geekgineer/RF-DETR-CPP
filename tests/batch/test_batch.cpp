// SPDX-FileCopyrightText: 2024 Geekgineer
// SPDX-License-Identifier: AGPL-3.0-only
//
// ============================================================================
// RFDETRs-CPP — Functional tests: batch image inference (library API)
// ============================================================================
// Exercises RFDETRDetector::detect() across multiple images in sequence,
// covering the same use-case as rfdetr_batch_inference without shelling out.
//
// Requires: tests/detection/models/rfdetr-base.onnx  (auto-skipped if absent)
//           tests/detection/data/images/dog.jpg
// ============================================================================

#include <gtest/gtest.h>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <filesystem>
#include <string>
#include <vector>

#include "rfdetrs/rfdetrs.hpp"

namespace fs = std::filesystem;
using namespace rfdetrs;
using namespace rfdetrs::det;

// BASE_PATH_DETECTION is injected by CMakeLists.txt
static const std::string MODEL_PATH = std::string(BASE_PATH_DETECTION) + "models/rfdetr-base.onnx";
static const std::string DOG_IMAGE  = std::string(BASE_PATH_DETECTION) + "data/images/dog.jpg";

// ============================================================================
// Fixture — loads the detector once for the suite; skips if model absent
// ============================================================================
class BatchFixture : public ::testing::Test {
protected:
    std::unique_ptr<RFDETRDetector> det_;

    void SetUp() override {
        if (!fs::exists(MODEL_PATH))
            GTEST_SKIP() << "Model not found (skipping functional tests): " << MODEL_PATH;
        det_ = std::make_unique<RFDETRDetector>(MODEL_PATH, "", BackendType::CPU);
    }
};

// ----------------------------------------------------------------------------
// Real-image tests
// ----------------------------------------------------------------------------

TEST_F(BatchFixture, RealImageProducesDetections) {
    cv::Mat img = cv::imread(DOG_IMAGE);
    ASSERT_FALSE(img.empty()) << "Cannot load: " << DOG_IMAGE;

    auto dets = det_->detect(img, 0.5f, 300);
    EXPECT_GT(dets.size(), 0u) << "Expected ≥1 detection on dog.jpg at conf=0.5";
}

TEST_F(BatchFixture, RealImageConfidenceRange) {
    cv::Mat img = cv::imread(DOG_IMAGE);
    ASSERT_FALSE(img.empty());

    for (const auto& d : det_->detect(img, 0.3f, 300)) {
        EXPECT_GT(d.conf, 0.0f) << "confidence ≤ 0";
        EXPECT_LE(d.conf, 1.0f) << "confidence > 1";
    }
}

TEST_F(BatchFixture, RealImageBBoxSanity) {
    cv::Mat img = cv::imread(DOG_IMAGE);
    ASSERT_FALSE(img.empty());

    for (const auto& d : det_->detect(img, 0.3f, 300)) {
        EXPECT_GE(d.box.x,      0);
        EXPECT_GE(d.box.y,      0);
        EXPECT_GT(d.box.width,  0);
        EXPECT_GT(d.box.height, 0);
        EXPECT_LE(d.box.x + d.box.width,  img.cols);
        EXPECT_LE(d.box.y + d.box.height, img.rows);
    }
}

TEST_F(BatchFixture, RealImageSortedDescending) {
    cv::Mat img = cv::imread(DOG_IMAGE);
    ASSERT_FALSE(img.empty());

    auto dets = det_->detect(img, 0.3f, 300);
    for (size_t i = 1; i < dets.size(); ++i)
        EXPECT_GE(dets[i - 1].conf, dets[i].conf - 1e-4f)
            << "Detections not sorted descending at index " << i;
}

TEST_F(BatchFixture, DeterministicOnSameInput) {
    cv::Mat img = cv::imread(DOG_IMAGE);
    ASSERT_FALSE(img.empty());

    auto d1 = det_->detect(img, 0.5f, 300);
    auto d2 = det_->detect(img, 0.5f, 300);

    ASSERT_EQ(d1.size(), d2.size()) << "Different count on identical inputs";
    for (size_t i = 0; i < d1.size(); ++i) {
        EXPECT_EQ(d1[i].classId,   d2[i].classId);
        EXPECT_FLOAT_EQ(d1[i].conf, d2[i].conf);
        EXPECT_EQ(d1[i].box.x,     d2[i].box.x);
        EXPECT_EQ(d1[i].box.y,     d2[i].box.y);
    }
}

// ----------------------------------------------------------------------------
// Edge-case / threshold tests
// ----------------------------------------------------------------------------

TEST_F(BatchFixture, TopKZeroReturnsEmpty) {
    cv::Mat img = cv::imread(DOG_IMAGE);
    ASSERT_FALSE(img.empty());
    EXPECT_EQ(det_->detect(img, 0.5f, 0).size(), 0u);
}

TEST_F(BatchFixture, ConfOneReturnsEmpty) {
    cv::Mat img = cv::imread(DOG_IMAGE);
    ASSERT_FALSE(img.empty());
    // Nothing can score above 100% after sigmoid
    EXPECT_EQ(det_->detect(img, 1.0f, 300).size(), 0u);
}

TEST_F(BatchFixture, BlankImageValidStructure) {
    // A uniform gray image will likely produce zero detections, but any that
    // do come back must still be structurally valid.
    cv::Mat blank(480, 640, CV_8UC3, cv::Scalar(128, 128, 128));
    for (const auto& d : det_->detect(blank, 0.5f, 300)) {
        EXPECT_GT(d.conf, 0.0f);
        EXPECT_LE(d.conf, 1.0f);
        EXPECT_GT(d.box.width,  0);
        EXPECT_GT(d.box.height, 0);
    }
}

TEST_F(BatchFixture, TopKLimitsResultCount) {
    cv::Mat img = cv::imread(DOG_IMAGE);
    ASSERT_FALSE(img.empty());

    constexpr int K = 3;
    auto dets = det_->detect(img, 0.0f, K);
    EXPECT_LE(static_cast<int>(dets.size()), K);
}

// ----------------------------------------------------------------------------
// Multi-image batch: sequential detect() calls — the core batch use-case
// ----------------------------------------------------------------------------

TEST_F(BatchFixture, MultiImageBatchValidStructure) {
    cv::Mat dog = cv::imread(DOG_IMAGE);
    ASSERT_FALSE(dog.empty());

    std::vector<cv::Mat> batch = {
        dog,
        cv::Mat(320, 320, CV_8UC3, cv::Scalar(  0,   0,   0)),
        cv::Mat(640, 480, CV_8UC3, cv::Scalar(200, 200, 200)),
        dog,
    };

    for (size_t i = 0; i < batch.size(); ++i) {
        auto dets = det_->detect(batch[i], 0.5f, 300);
        for (const auto& d : dets) {
            EXPECT_GT(d.conf, 0.0f) << "image[" << i << "]: conf ≤ 0";
            EXPECT_LE(d.conf, 1.0f) << "image[" << i << "]: conf > 1";
            EXPECT_GT(d.box.width,  0) << "image[" << i << "]: non-positive width";
            EXPECT_GT(d.box.height, 0) << "image[" << i << "]: non-positive height";
            EXPECT_LE(d.box.x + d.box.width,  batch[i].cols) << "image[" << i << "]: box overflows width";
            EXPECT_LE(d.box.y + d.box.height, batch[i].rows) << "image[" << i << "]: box overflows height";
        }
    }
}

TEST_F(BatchFixture, SameImageFirstAndLastMatch) {
    cv::Mat img = cv::imread(DOG_IMAGE);
    ASSERT_FALSE(img.empty());

    // dog is at index 0 and 3; their results must be identical
    cv::Mat blank1(320, 320, CV_8UC3, cv::Scalar(0,   0,   0));
    cv::Mat blank2(640, 480, CV_8UC3, cv::Scalar(200, 200, 200));

    auto d0 = det_->detect(img,    0.5f, 300);
    auto _1 = det_->detect(blank1, 0.5f, 300);
    auto _2 = det_->detect(blank2, 0.5f, 300);
    auto d3 = det_->detect(img,    0.5f, 300);

    ASSERT_EQ(d0.size(), d3.size()) << "Same image at idx 0 and 3 produced different count";
    for (size_t i = 0; i < d0.size(); ++i) {
        EXPECT_EQ(d0[i].classId, d3[i].classId);
        EXPECT_FLOAT_EQ(d0[i].conf, d3[i].conf);
    }
    (void)_1; (void)_2;
}
