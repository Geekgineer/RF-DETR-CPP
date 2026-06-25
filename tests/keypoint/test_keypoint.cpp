// SPDX-FileCopyrightText: 2024 Geekgineer
// SPDX-License-Identifier: AGPL-3.0-only
//
// ============================================================================
// RFDETRs-CPP — Structural tests: keypoint inference (library API)
// ============================================================================
// Exercises RFDETRKeypointDetector on structural and edge-case inputs.
// All tests auto-skip when the keypoint model is absent so CI never fails
// due to missing model files.
//
// Requires: tests/keypoint/models/*.onnx  (any rfdetr-keypoint-*.onnx)
// For the COCO preview model pass kpsPerClass = {0, 17}.
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
using namespace rfdetrs::kp;

static const std::string DOG_IMAGE  = std::string(BASE_PATH_DETECTION) + "data/images/dog.jpg";
static const std::string KP_MODEL_DIR = std::string(BASE_PATH_KEYPOINT) + "models/";

// COCO KeypointPreview: class 0 = background (0 kps), class 1 = person (17 kps).
static const std::vector<int> COCO_KPS_PER_CLASS = {0, 17};

static std::string findKpModel() {
    if (!fs::is_directory(KP_MODEL_DIR)) return "";
    for (const auto& e : fs::directory_iterator(KP_MODEL_DIR)) {
        if (e.path().extension() == ".onnx") return e.path().string();
    }
    return "";
}

// ============================================================================
// Fixture
// ============================================================================
class KeypointFixture : public ::testing::Test {
protected:
    std::unique_ptr<RFDETRKeypointDetector> det_;
    std::string modelPath_;

    void SetUp() override {
        modelPath_ = findKpModel();
        if (modelPath_.empty())
            GTEST_SKIP() << "No keypoint model found in: " << KP_MODEL_DIR;
        // Default to COCO schema; adjust if a different model is used.
        det_ = std::make_unique<RFDETRKeypointDetector>(
            modelPath_, "", COCO_KPS_PER_CLASS, BackendType::CPU);
    }
};

// ============================================================================
// Structural tests
// ============================================================================

TEST_F(KeypointFixture, BlankImageValidStructure) {
    cv::Mat blank(480, 640, CV_8UC3, cv::Scalar(128, 128, 128));
    auto dets = det_->detect(blank, 0.5f, 300);
    for (const auto& d : dets) {
        EXPECT_GT(d.conf, 0.0f);
        EXPECT_LE(d.conf, 1.0f);
        EXPECT_GT(d.box.width,  0);
        EXPECT_GT(d.box.height, 0);
        // Keypoints must be non-empty (may all have low confidence on blank)
        EXPECT_FALSE(d.keypoints.empty());
        for (const auto& kp : d.keypoints) {
            EXPECT_GE(kp.x,          0.f);
            EXPECT_LE(kp.x, static_cast<float>(blank.cols));
            EXPECT_GE(kp.y,          0.f);
            EXPECT_LE(kp.y, static_cast<float>(blank.rows));
            EXPECT_GE(kp.confidence, 0.f);
            EXPECT_LE(kp.confidence, 1.f);
            EXPECT_GE(kp.visible,    0.f);
            EXPECT_LE(kp.visible,    1.f);
        }
    }
}

TEST_F(KeypointFixture, ConfidenceInRange) {
    cv::Mat img = cv::imread(DOG_IMAGE);
    if (img.empty()) GTEST_SKIP() << "dog.jpg not available";

    for (const auto& d : det_->detect(img, 0.3f, 300)) {
        EXPECT_GT(d.conf, 0.0f);
        EXPECT_LE(d.conf, 1.0f);
    }
}

TEST_F(KeypointFixture, BBoxSanity) {
    cv::Mat img = cv::imread(DOG_IMAGE);
    if (img.empty()) GTEST_SKIP() << "dog.jpg not available";

    for (const auto& d : det_->detect(img, 0.3f, 300)) {
        EXPECT_GE(d.box.x, 0);
        EXPECT_GE(d.box.y, 0);
        EXPECT_GT(d.box.width,  0);
        EXPECT_GT(d.box.height, 0);
        EXPECT_LE(d.box.x + d.box.width,  img.cols);
        EXPECT_LE(d.box.y + d.box.height, img.rows);
    }
}

TEST_F(KeypointFixture, KeypointCountMatchesSchema) {
    cv::Mat img = cv::imread(DOG_IMAGE);
    if (img.empty()) GTEST_SKIP() << "dog.jpg not available";

    auto dets = det_->detect(img, 0.3f, 300);
    for (const auto& d : dets) {
        if (d.classId < 0) continue;
        if (d.classId < static_cast<int>(COCO_KPS_PER_CLASS.size())) {
            // Class is within schema — keypoint count must match exactly.
            int expectedKps = COCO_KPS_PER_CLASS[static_cast<size_t>(d.classId)];
            EXPECT_EQ(static_cast<int>(d.keypoints.size()), expectedKps)
                << "classId=" << d.classId << " should have " << expectedKps << " keypoints";
        } else {
            // Class beyond schema — no keypoints expected.
            EXPECT_EQ(d.keypoints.size(), 0u)
                << "classId=" << d.classId << " is beyond schema and should have 0 keypoints";
        }
    }
}

TEST_F(KeypointFixture, KeypointCoordinatesWithinImage) {
    cv::Mat img = cv::imread(DOG_IMAGE);
    if (img.empty()) GTEST_SKIP() << "dog.jpg not available";

    auto dets = det_->detect(img, 0.3f, 300);
    for (const auto& d : dets) {
        for (size_t ki = 0; ki < d.keypoints.size(); ++ki) {
            const auto& kp = d.keypoints[ki];
            EXPECT_GE(kp.x, 0.f)                          << "kp[" << ki << "].x < 0";
            EXPECT_LE(kp.x, static_cast<float>(img.cols)) << "kp[" << ki << "].x > width";
            EXPECT_GE(kp.y, 0.f)                          << "kp[" << ki << "].y < 0";
            EXPECT_LE(kp.y, static_cast<float>(img.rows)) << "kp[" << ki << "].y > height";
        }
    }
}

TEST_F(KeypointFixture, KeypointConfidenceInRange) {
    cv::Mat img = cv::imread(DOG_IMAGE);
    if (img.empty()) GTEST_SKIP() << "dog.jpg not available";

    for (const auto& d : det_->detect(img, 0.3f, 300)) {
        for (const auto& kp : d.keypoints) {
            EXPECT_GE(kp.confidence, 0.f);
            EXPECT_LE(kp.confidence, 1.f);
            EXPECT_GE(kp.visible,    0.f);
            EXPECT_LE(kp.visible,    1.f);
        }
    }
}

TEST_F(KeypointFixture, TopKZeroReturnsEmpty) {
    cv::Mat img = cv::imread(DOG_IMAGE);
    if (img.empty()) GTEST_SKIP() << "dog.jpg not available";
    EXPECT_EQ(det_->detect(img, 0.5f, 0).size(), 0u);
}

TEST_F(KeypointFixture, ConfOneReturnsEmpty) {
    cv::Mat img = cv::imread(DOG_IMAGE);
    if (img.empty()) GTEST_SKIP() << "dog.jpg not available";
    EXPECT_EQ(det_->detect(img, 1.0f, 300).size(), 0u);
}

TEST_F(KeypointFixture, TopKLimitsResultCount) {
    cv::Mat img = cv::imread(DOG_IMAGE);
    if (img.empty()) GTEST_SKIP() << "dog.jpg not available";
    constexpr int K = 3;
    EXPECT_LE(static_cast<int>(det_->detect(img, 0.0f, K).size()), K);
}

TEST_F(KeypointFixture, SortedDescending) {
    cv::Mat img = cv::imread(DOG_IMAGE);
    if (img.empty()) GTEST_SKIP() << "dog.jpg not available";

    auto dets = det_->detect(img, 0.3f, 300);
    for (size_t i = 1; i < dets.size(); ++i)
        EXPECT_GE(dets[i - 1].conf, dets[i].conf - 1e-4f)
            << "Detections not sorted descending at index " << i;
}

TEST_F(KeypointFixture, Deterministic) {
    cv::Mat img = cv::imread(DOG_IMAGE);
    if (img.empty()) GTEST_SKIP() << "dog.jpg not available";

    auto d1 = det_->detect(img, 0.5f, 300);
    auto d2 = det_->detect(img, 0.5f, 300);
    ASSERT_EQ(d1.size(), d2.size());
    for (size_t i = 0; i < d1.size(); ++i) {
        EXPECT_EQ(d1[i].classId, d2[i].classId);
        EXPECT_FLOAT_EQ(d1[i].conf, d2[i].conf);
    }
}

TEST_F(KeypointFixture, VaryingImageSizesNoCrash) {
    std::vector<cv::Size> sizes = {{320, 240}, {640, 480}, {160, 120}};
    for (const auto& sz : sizes) {
        cv::Mat frame(sz.height, sz.width, CV_8UC3, cv::Scalar(100, 100, 100));
        EXPECT_NO_THROW(det_->detect(frame, 0.5f, 300)) << "crash at size " << sz;
    }
}

TEST_F(KeypointFixture, TooLongSchemaThrows) {
    if (modelPath_.empty()) GTEST_SKIP() << "No model available";
    // Schema longer than numClasses should throw (cannot have more schema classes than ONNX classes)
    std::vector<int> tooLong(200, 17);
    EXPECT_THROW(
        RFDETRKeypointDetector bad(modelPath_, "", tooLong, BackendType::CPU),
        std::runtime_error
    ) << "Schema longer than numClasses should throw";
}

// Verify a detection model is rejected
TEST_F(KeypointFixture, DetectionModelRejected) {
    const std::string detModel = std::string(BASE_PATH_DETECTION) + "models/rfdetr-base.onnx";
    if (!fs::exists(detModel))
        GTEST_SKIP() << "Detection model not available for cross-check";
    EXPECT_THROW(
        RFDETRKeypointDetector bad(detModel, "", {}, BackendType::CPU),
        std::runtime_error
    ) << "Loading a detection model into RFDETRKeypointDetector should throw";
}
