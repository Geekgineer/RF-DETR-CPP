// SPDX-FileCopyrightText: 2024 Geekgineer
// SPDX-License-Identifier: AGPL-3.0-only
//
// ============================================================================
// RFDETRs-CPP — Live camera inference
// ============================================================================
// Captures frames from a camera device, runs RF-DETR detection, and displays
// results in real time using a producer-consumer pipeline with double buffering.
//
// Usage:
//   rfdetr_camera_inference <model.onnx> [camera=/dev/video0|0]
//                           [labels.names] [conf=0.5] [top_k=300]
//                           [backend=cpu|cuda|trt]
//
// Controls: press 'q' to quit, 's' to save a snapshot.
// ============================================================================

#include <atomic>
#include <chrono>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>

#include "rfdetrs/rfdetrs.hpp"
#include "rfdetrs/tools/BoundedThreadSafeQueue.hpp"

using namespace rfdetrs;
using namespace rfdetrs::det;
namespace fs = std::filesystem;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
static BackendType parseBackend(const std::string& s) {
    if (s == "cuda") return BackendType::CUDA;
    if (s == "trt")  return BackendType::TensorRT;
    return BackendType::CPU;
}

static std::string timestamp() {
    auto now    = std::chrono::system_clock::now();
    auto now_t  = std::chrono::system_clock::to_time_t(now);
    std::tm tm{};
    localtime_r(&now_t, &tm);
    std::ostringstream oss;
    oss << std::put_time(&tm, "%Y%m%d_%H%M%S");
    return oss.str();
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0]
                  << " <model.onnx> [camera=/dev/video0|0]"
                  << " [labels.names] [conf=0.5] [top_k=300] [backend=cpu|cuda|trt]\n";
        return 1;
    }

    std::string modelPath   = argv[1];
    std::string videoSource = (argc > 2) ? argv[2] : "/dev/video0";
    std::string labelsPath  = (argc > 3) ? argv[3] : "";
    float conf              = (argc > 4) ? std::stof(argv[4]) : 0.5f;
    int   topK              = (argc > 5) ? std::stoi(argv[5]) : 300;
    BackendType backend     = (argc > 6) ? parseBackend(argv[6]) : BackendType::CPU;

    std::cout << "[INFO] Model:  " << modelPath   << "\n";
    std::cout << "[INFO] Camera: " << videoSource << "\n";
    std::cout << "[INFO] Conf:   " << conf << "  TopK: " << topK << "\n";
    std::cout << "[INFO] Press 'q' to quit, 's' to save snapshot\n";

    RFDETRDetector detector(modelPath, labelsPath, backend);

    cv::VideoCapture cap;
    // Try numeric camera index first, then device path
    bool isNumeric = !videoSource.empty() &&
        videoSource.find_first_not_of("0123456789") == std::string::npos;
    if (isNumeric)
        cap.open(std::stoi(videoSource));
    else
        cap.open(videoSource, cv::CAP_V4L2);

    if (!cap.isOpened()) {
        std::cerr << "[ERROR] Cannot open camera: " << videoSource << "\n";
        return 1;
    }
    cap.set(cv::CAP_PROP_FRAME_WIDTH,  1280);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 720);
    cap.set(cv::CAP_PROP_FPS, 30);

    const size_t queueCap = 2;  // double buffering
    BoundedThreadSafeQueue<cv::Mat> frameQueue(queueCap);
    BoundedThreadSafeQueue<std::pair<cv::Mat, std::vector<Detection>>> processedQueue(queueCap);
    std::atomic<bool> stop{false};

    // Producer: capture frames
    std::thread producer([&] {
        cv::Mat frame;
        while (!stop.load() && cap.read(frame))
            if (!frameQueue.enqueue(frame.clone())) break;
        frameQueue.set_finished();
    });

    // Consumer: run detection
    std::thread consumer([&] {
        cv::Mat frame;
        while (!stop.load() && frameQueue.dequeue(frame)) {
            auto dets = detector.detect(frame, conf, topK);
            if (!processedQueue.enqueue({frame, dets})) break;
        }
        processedQueue.set_finished();
    });

    // Display on main thread (required on macOS; fine on Linux too)
    int frameCount = 0;
    auto startTime = std::chrono::steady_clock::now();
    std::pair<cv::Mat, std::vector<Detection>> item;
    fs::create_directories("outputs/det");

    while (!stop.load() && processedQueue.dequeue(item)) {
        cv::Mat& disp = item.first;
        detector.drawDetections(disp, item.second);

        // FPS overlay
        ++frameCount;
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::steady_clock::now() - startTime).count();
        double fps = (elapsed > 0) ? frameCount / static_cast<double>(elapsed) : 0.0;
        std::string overlay = "FPS: " + std::to_string(static_cast<int>(fps)) +
                              "  Dets: " + std::to_string(item.second.size());
        cv::putText(disp, overlay, {10, 30},
                    cv::FONT_HERSHEY_SIMPLEX, 0.8, {0, 255, 0}, 2);

        cv::imshow("RF-DETR Camera (q=quit, s=snapshot)", disp);
        char key = static_cast<char>(cv::waitKey(1));
        if (key == 'q') {
            stop.store(true);
            frameQueue.set_finished();
            processedQueue.set_finished();
        } else if (key == 's') {
            std::string path = "outputs/det/snapshot_" + timestamp() + ".jpg";
            cv::imwrite(path, disp);
            std::cout << "[INFO] Snapshot saved: " << path << "\n";
        }
    }

    producer.join();
    consumer.join();
    cap.release();
    cv::destroyAllWindows();

    std::cout << "[INFO] Processed " << frameCount << " frames.\n";
    return 0;
}
