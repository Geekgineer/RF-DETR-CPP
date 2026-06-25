// SPDX-FileCopyrightText: 2024 Geekgineer
// SPDX-License-Identifier: AGPL-3.0-only
//
// ============================================================================
// RFDETRs-CPP — Structural tests: segmentation inference (library API)
// ============================================================================
// Exercises RFDETRSegDetector on structural and edge-case inputs.
// All tests auto-skip when the segmentation model is absent so CI never fails
// due to missing model files.
//
// Requires: tests/segmentation/models/*.onnx  (any rfdetr-seg-*.onnx)
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
using namespace rfdetrs::seg;

// BASE_PATH_DETECTION is injected by CMakeLists.txt (detection test tree)
static const std::string DOG_IMAGE  = std::string(BASE_PATH_DETECTION) + "data/images/dog.jpg";

// Segmentation models live in the seg test directory.
static const std::string SEG_MODEL_DIR = std::string(BASE_PATH_SEGMENTATION) + "models/";

// Find the first .onnx in the segmentation model directory.
static std::string findSegModel() {
    if (!fs::is_directory(SEG_MODEL_DIR)) return "";
    for (const auto& e : fs::directory_iterator(SEG_MODEL_DIR)) {
        if (e.path().extension() == ".onnx") return e.path().string();
    }
    return "";
}

// ============================================================================
// Fixture — loads the segmentation detector once per suite
// ============================================================================
class SegFixture : public ::testing::Test {
protected:
    std::unique_ptr<RFDETRSegDetector> det_;
    std::string modelPath_;

    void SetUp() override {
        modelPath_ = findSegModel();
        if (modelPath_.empty())
            GTEST_SKIP() << "No segmentation model found in: " << SEG_MODEL_DIR;
        det_ = std::make_unique<RFDETRSegDetector>(modelPath_, "", BackendType::CPU);
    }
};

// ============================================================================
// Structural validity — any detection must have valid fields
// ============================================================================

TEST_F(SegFixture, BlankImageValidStructure) {
    cv::Mat blank(480, 640, CV_8UC3, cv::Scalar(128, 128, 128));
    auto dets = det_->detect(blank, 0.5f, 300);
    for (const auto& d : dets) {
        EXPECT_GT(d.conf, 0.0f);
        EXPECT_LE(d.conf, 1.0f);
        EXPECT_GT(d.box.width,  0);
        EXPECT_GT(d.box.height, 0);
        EXPECT_GE(d.box.x, 0);
        EXPECT_GE(d.box.y, 0);
        // Mask must be CV_8UC1 and same size as image
        ASSERT_FALSE(d.mask.empty());
        EXPECT_EQ(d.mask.type(), CV_8UC1);
        EXPECT_EQ(d.mask.cols, blank.cols);
        EXPECT_EQ(d.mask.rows, blank.rows);
    }
}

TEST_F(SegFixture, ConfidenceInRange) {
    cv::Mat img = cv::imread(DOG_IMAGE);
    if (img.empty()) GTEST_SKIP() << "dog.jpg not available";

    for (const auto& d : det_->detect(img, 0.3f, 300)) {
        EXPECT_GT(d.conf, 0.0f);
        EXPECT_LE(d.conf, 1.0f);
    }
}

TEST_F(SegFixture, MaskSizeMatchesImage) {
    cv::Mat img = cv::imread(DOG_IMAGE);
    if (img.empty()) GTEST_SKIP() << "dog.jpg not available";

    auto dets = det_->detect(img, 0.3f, 300);
    for (const auto& d : dets) {
        ASSERT_FALSE(d.mask.empty()) << "mask is empty for detection";
        EXPECT_EQ(d.mask.cols, img.cols) << "mask width mismatch";
        EXPECT_EQ(d.mask.rows, img.rows) << "mask height mismatch";
        EXPECT_EQ(d.mask.type(), CV_8UC1) << "mask must be CV_8UC1";
    }
}

TEST_F(SegFixture, MaskIsBinary) {
    cv::Mat img = cv::imread(DOG_IMAGE);
    if (img.empty()) GTEST_SKIP() << "dog.jpg not available";

    auto dets = det_->detect(img, 0.3f, 300);
    for (const auto& d : dets) {
        if (d.mask.empty()) continue;
        double minVal, maxVal;
        cv::minMaxLoc(d.mask, &minVal, &maxVal);
        EXPECT_EQ(static_cast<int>(minVal), 0)   << "mask minimum must be 0";
        EXPECT_EQ(static_cast<int>(maxVal), 255) << "mask maximum must be 255 (binary)";
    }
}

TEST_F(SegFixture, BBoxSanity) {
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

TEST_F(SegFixture, TopKLimitsResultCount) {
    cv::Mat img = cv::imread(DOG_IMAGE);
    if (img.empty()) GTEST_SKIP() << "dog.jpg not available";

    constexpr int K = 3;
    auto dets = det_->detect(img, 0.0f, K);
    EXPECT_LE(static_cast<int>(dets.size()), K);
}

TEST_F(SegFixture, TopKZeroReturnsEmpty) {
    cv::Mat img = cv::imread(DOG_IMAGE);
    if (img.empty()) GTEST_SKIP() << "dog.jpg not available";
    EXPECT_EQ(det_->detect(img, 0.5f, 0).size(), 0u);
}

TEST_F(SegFixture, ConfOneReturnsEmpty) {
    cv::Mat img = cv::imread(DOG_IMAGE);
    if (img.empty()) GTEST_SKIP() << "dog.jpg not available";
    EXPECT_EQ(det_->detect(img, 1.0f, 300).size(), 0u);
}

TEST_F(SegFixture, SortedDescending) {
    cv::Mat img = cv::imread(DOG_IMAGE);
    if (img.empty()) GTEST_SKIP() << "dog.jpg not available";

    auto dets = det_->detect(img, 0.3f, 300);
    for (size_t i = 1; i < dets.size(); ++i)
        EXPECT_GE(dets[i - 1].conf, dets[i].conf - 1e-4f)
            << "Detections not sorted descending at index " << i;
}

TEST_F(SegFixture, Deterministic) {
    cv::Mat img = cv::imread(DOG_IMAGE);
    if (img.empty()) GTEST_SKIP() << "dog.jpg not available";

    auto d1 = det_->detect(img, 0.5f, 300);
    auto d2 = det_->detect(img, 0.5f, 300);
    ASSERT_EQ(d1.size(), d2.size()) << "Non-deterministic across identical calls";
    for (size_t i = 0; i < d1.size(); ++i) {
        EXPECT_EQ(d1[i].classId, d2[i].classId);
        EXPECT_FLOAT_EQ(d1[i].conf,   d2[i].conf);
        EXPECT_EQ(d1[i].box.x,   d2[i].box.x);
        EXPECT_EQ(d1[i].box.y,   d2[i].box.y);
    }
}

TEST_F(SegFixture, VaryingImageSizesNoCrash) {
    std::vector<cv::Size> sizes = {{320, 240}, {640, 480}, {160, 120}, {1280, 720}};
    for (const auto& sz : sizes) {
        cv::Mat frame(sz.height, sz.width, CV_8UC3, cv::Scalar(100, 100, 100));
        EXPECT_NO_THROW(det_->detect(frame, 0.5f, 300)) << "crash at size " << sz;
        auto dets = det_->detect(frame, 0.5f, 300);
        for (const auto& d : dets) {
            EXPECT_EQ(d.mask.cols, sz.width)  << "mask width mismatch at size " << sz;
            EXPECT_EQ(d.mask.rows, sz.height) << "mask height mismatch at size " << sz;
        }
    }
}

// Verify the model correctly identifies itself as a seg model by rejecting a detection model
TEST_F(SegFixture, DetectionModelRejected) {
    const std::string detModel = std::string(BASE_PATH_DETECTION) + "models/rfdetr-base.onnx";
    if (!fs::exists(detModel))
        GTEST_SKIP() << "Detection model not available for cross-check";
    EXPECT_THROW(
        RFDETRSegDetector bad(detModel, "", BackendType::CPU),
        std::runtime_error
    ) << "Loading a detection model into RFDETRSegDetector should throw";
}
