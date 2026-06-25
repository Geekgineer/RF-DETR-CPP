// SPDX-FileCopyrightText: 2024 Geekgineer
// SPDX-License-Identifier: AGPL-3.0-only
//
// ============================================================================
// RFDETRs-CPP — Video inference
// ============================================================================
// Reads a video file, runs RF-DETR detection on each frame using a
// threaded capture → process → write pipeline, and saves an annotated video.
//
// Usage:
//   rfdetr_video_inference <model.onnx> <input.mp4> [output.mp4]
//                          [labels.names] [conf=0.5] [top_k=300]
//                          [backend=cpu|cuda|trt]
// ============================================================================

#include <atomic>
#include <condition_variable>
#include <iostream>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <utility>

#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>

#include "rfdetrs/rfdetrs.hpp"

using namespace rfdetrs;
using namespace rfdetrs::det;

// ---------------------------------------------------------------------------
// Unbounded thread-safe queue (capture → process → write)
// ---------------------------------------------------------------------------
template <typename T>
class SafeQueue {
public:
    void enqueue(T t) {
        std::lock_guard<std::mutex> lock(mutex_);
        queue_.push(std::move(t));
        cv_.notify_one();
    }

    bool dequeue(T& t) {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [this] { return !queue_.empty() || finished_; });
        if (queue_.empty()) return false;
        t = std::move(queue_.front());
        queue_.pop();
        return true;
    }

    void setFinished() {
        std::lock_guard<std::mutex> lock(mutex_);
        finished_ = true;
        cv_.notify_all();
    }

private:
    std::queue<T> queue_;
    std::mutex mutex_;
    std::condition_variable cv_;
    bool finished_{false};
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
static BackendType parseBackend(const std::string& s) {
    if (s == "cuda") return BackendType::CUDA;
    if (s == "trt")  return BackendType::TensorRT;
    return BackendType::CPU;
}

static std::string defaultOutput(const std::string& inputPath) {
    size_t dot = inputPath.rfind('.');
    std::string base = (dot == std::string::npos) ? inputPath : inputPath.substr(0, dot);
    return base + "_rfdetr.mp4";
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0]
                  << " <model.onnx> <input.mp4>"
                  << " [output.mp4] [labels.names]"
                  << " [conf=0.5] [top_k=300] [backend=cpu|cuda|trt]\n";
        return 1;
    }

    std::string modelPath  = argv[1];
    std::string videoPath  = argv[2];
    std::string outputPath = (argc > 3) ? argv[3] : defaultOutput(videoPath);
    std::string labelsPath = (argc > 4) ? argv[4] : "";
    float conf             = (argc > 5) ? std::stof(argv[5]) : 0.5f;
    int   topK             = (argc > 6) ? std::stoi(argv[6]) : 300;
    BackendType backend    = (argc > 7) ? parseBackend(argv[7]) : BackendType::CPU;

    std::cout << "[INFO] Model:   " << modelPath  << "\n";
    std::cout << "[INFO] Input:   " << videoPath  << "\n";
    std::cout << "[INFO] Output:  " << outputPath << "\n";
    std::cout << "[INFO] Conf:    " << conf << "  TopK: " << topK << "\n";

    RFDETRDetector detector(modelPath, labelsPath, backend);

    cv::VideoCapture cap(videoPath);
    if (!cap.isOpened()) {
        std::cerr << "[ERROR] Cannot open video: " << videoPath << "\n";
        return 1;
    }

    const int    frameW  = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    const int    frameH  = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    const double fps     = cap.get(cv::CAP_PROP_FPS);
    const int    fourcc  = cv::VideoWriter::fourcc('m', 'p', '4', 'v');

    cv::VideoWriter writer(outputPath, fourcc, fps, cv::Size(frameW, frameH), true);
    if (!writer.isOpened()) {
        std::cerr << "[ERROR] Cannot open video writer: " << outputPath << "\n";
        return 1;
    }

    SafeQueue<cv::Mat>                   frameQueue;
    SafeQueue<std::pair<int, cv::Mat>>   processedQueue;

    // Capture thread
    std::thread captureThread([&] {
        cv::Mat frame;
        while (cap.read(frame))
            frameQueue.enqueue(frame.clone());
        frameQueue.setFinished();
    });

    // Processing thread
    std::thread processingThread([&] {
        cv::Mat frame;
        int idx = 0;
        while (frameQueue.dequeue(frame)) {
            auto dets = detector.detect(frame, conf, topK);
            detector.drawDetections(frame, dets);
            if (idx % 30 == 0)
                std::cout << "[FRAME " << idx << "] detections=" << dets.size() << "\n";
            processedQueue.enqueue(std::make_pair(idx++, frame));
        }
        processedQueue.setFinished();
    });

    // Writing thread
    std::thread writingThread([&] {
        std::pair<int, cv::Mat> item;
        while (processedQueue.dequeue(item))
            writer.write(item.second);
    });

    captureThread.join();
    processingThread.join();
    writingThread.join();

    cap.release();
    writer.release();
    cv::destroyAllWindows();

    std::cout << "[INFO] Saved annotated video: " << outputPath << "\n";
    return 0;
}
