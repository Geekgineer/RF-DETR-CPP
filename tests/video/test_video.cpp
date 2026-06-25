// SPDX-FileCopyrightText: 2024 Geekgineer
// SPDX-License-Identifier: AGPL-3.0-only
//
// ============================================================================
// RFDETRs-CPP — Functional tests: video inference (library API)
// ============================================================================
// Exercises RFDETRDetector::detect() frame-by-frame over an OpenCV VideoCapture,
// covering the same use-case as rfdetr_video_inference without shelling out.
//
// SetUp creates a synthetic 10-frame MP4 (some frames are a resized dog.jpg,
// others are solid-colour blanks) so the tests run without a real camera.
// TearDown removes the temp file.
//
// Requires: tests/detection/models/rfdetr-base.onnx  (auto-skipped if absent)
// ============================================================================

#include <gtest/gtest.h>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

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
// Fixture — creates a synthetic video before each test; removes it after
// ============================================================================
class VideoFixture : public ::testing::Test {
protected:
    fs::path videoPath_;
    std::unique_ptr<RFDETRDetector> det_;

    static constexpr int FRAMES = 10;
    static constexpr int FPS    = 5;
    static constexpr int W      = 320;
    static constexpr int H      = 240;

    void SetUp() override {
        if (!fs::exists(MODEL_PATH))
            GTEST_SKIP() << "Model not found (skipping functional tests): " << MODEL_PATH;

        videoPath_ = fs::temp_directory_path() / "rfdetrs_test_video.mp4";

        {
            cv::VideoWriter writer(videoPath_.string(),
                                   cv::VideoWriter::fourcc('m','p','4','v'),
                                   FPS, cv::Size(W, H), true);
            if (!writer.isOpened())
                GTEST_SKIP() << "cv::VideoWriter cannot create MP4 (missing codec?)";

            cv::Mat dog = cv::imread(DOG_IMAGE);
            cv::Mat dogResized;
            if (!dog.empty())
                cv::resize(dog, dogResized, cv::Size(W, H));

            for (int i = 0; i < FRAMES; ++i) {
                if (!dogResized.empty() && i % 2 == 0) {
                    writer.write(dogResized);
                } else {
                    // Alternate hue so consecutive blank frames differ
                    int v = 80 + (i * 20) % 80;
                    writer.write(cv::Mat(H, W, CV_8UC3, cv::Scalar(v, v + 20, v + 40)));
                }
            }
            writer.release();
        }

        det_ = std::make_unique<RFDETRDetector>(MODEL_PATH, "", BackendType::CPU);
    }

    void TearDown() override {
        fs::remove(videoPath_);
    }
};

// ----------------------------------------------------------------------------
// Video I/O tests
// ----------------------------------------------------------------------------

TEST_F(VideoFixture, VideoFileIsReadable) {
    cv::VideoCapture cap(videoPath_.string());
    ASSERT_TRUE(cap.isOpened()) << "Cannot open synthetic test video";
    EXPECT_GT(static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT)), 0);
    EXPECT_GT(cap.get(cv::CAP_PROP_FPS), 0.0);
}

TEST_F(VideoFixture, FrameSizeMatchesWriter) {
    cv::VideoCapture cap(videoPath_.string());
    ASSERT_TRUE(cap.isOpened());
    cv::Mat frame;
    ASSERT_TRUE(cap.read(frame));
    EXPECT_EQ(frame.cols, W);
    EXPECT_EQ(frame.rows, H);
}

TEST_F(VideoFixture, AllFramesReadable) {
    cv::VideoCapture cap(videoPath_.string());
    ASSERT_TRUE(cap.isOpened());
    cv::Mat frame;
    int count = 0;
    while (cap.read(frame)) {
        ASSERT_FALSE(frame.empty()) << "Got empty frame at index " << count;
        ++count;
    }
    EXPECT_EQ(count, FRAMES) << "Expected " << FRAMES << " frames, got " << count;
}

// ----------------------------------------------------------------------------
// Detection correctness tests
// ----------------------------------------------------------------------------

TEST_F(VideoFixture, PerFrameDetectionsAreValid) {
    cv::VideoCapture cap(videoPath_.string());
    ASSERT_TRUE(cap.isOpened());

    cv::Mat frame;
    int idx = 0;
    while (cap.read(frame)) {
        auto dets = det_->detect(frame, 0.3f, 300);
        for (const auto& d : dets) {
            EXPECT_GT(d.conf, 0.0f) << "frame=" << idx << ": conf ≤ 0";
            EXPECT_LE(d.conf, 1.0f) << "frame=" << idx << ": conf > 1";
            EXPECT_GE(d.box.x,      0) << "frame=" << idx << ": box.x < 0";
            EXPECT_GE(d.box.y,      0) << "frame=" << idx << ": box.y < 0";
            EXPECT_GT(d.box.width,  0) << "frame=" << idx << ": non-positive width";
            EXPECT_GT(d.box.height, 0) << "frame=" << idx << ": non-positive height";
            EXPECT_LE(d.box.x + d.box.width,  frame.cols) << "frame=" << idx << ": box overflows width";
            EXPECT_LE(d.box.y + d.box.height, frame.rows) << "frame=" << idx << ": box overflows height";
        }
        ++idx;
    }
    EXPECT_EQ(idx, FRAMES);
}

TEST_F(VideoFixture, PerFrameDetectionsSortedDescending) {
    cv::VideoCapture cap(videoPath_.string());
    ASSERT_TRUE(cap.isOpened());

    cv::Mat frame;
    int idx = 0;
    while (cap.read(frame)) {
        auto dets = det_->detect(frame, 0.3f, 300);
        for (size_t i = 1; i < dets.size(); ++i)
            EXPECT_GE(dets[i - 1].conf, dets[i].conf - 1e-4f)
                << "frame=" << idx << ": detections not sorted descending at i=" << i;
        ++idx;
    }
}

TEST_F(VideoFixture, ConsecutiveFramesNoCrash) {
    cv::VideoCapture cap(videoPath_.string());
    ASSERT_TRUE(cap.isOpened());
    cv::Mat frame;
    while (cap.read(frame))
        EXPECT_NO_THROW(det_->detect(frame, 0.5f, 300));
}

// thread_local scored buffer must handle different frame sizes without stale data.
TEST_F(VideoFixture, VaryingFrameSizesNoCrash) {
    std::vector<cv::Size> sizes = {{320, 240}, {640, 480}, {160, 120}};
    for (const auto& sz : sizes) {
        cv::Mat frame(sz.height, sz.width, CV_8UC3, cv::Scalar(100, 100, 100));
        EXPECT_NO_THROW(det_->detect(frame, 0.5f, 300)) << "crash at size " << sz;
    }
}

// Dog frames (even indices in our synthetic video) should each produce
// at least as many detections as blank frames.
TEST_F(VideoFixture, DogFramesHaveMoreOrEqualDetectionsThanBlank) {
    cv::VideoCapture cap(videoPath_.string());
    ASSERT_TRUE(cap.isOpened());

    cv::Mat frame;
    int idx = 0;
    size_t dogTotal = 0, blankTotal = 0;
    while (cap.read(frame)) {
        auto dets = det_->detect(frame, 0.5f, 300);
        if (idx % 2 == 0) dogTotal  += dets.size();
        else               blankTotal += dets.size();
        ++idx;
    }
    // Dog frames should collectively yield at least as many hits as blank frames.
    EXPECT_GE(dogTotal, blankTotal)
        << "Blank frames produced more detections than dog frames — preprocessing suspect";
}
