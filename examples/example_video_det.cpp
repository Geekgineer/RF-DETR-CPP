// ============================================================================
// RFDETRs-CPP — Example: video detection
// ============================================================================
// Processes a video file frame-by-frame with RF-DETR, writes annotated output.
//
// Usage: example_video_det [model.onnx] [video.mp4] [labels.names]
// ============================================================================

#include <chrono>
#include <filesystem>
#include <iostream>
#include <string>

#include <opencv2/opencv.hpp>

#include "rfdetrs/rfdetrs.hpp"
#include "utils.hpp"

using namespace rfdetrs;
using namespace rfdetrs::det;

int main(int argc, char* argv[]) {
    std::string modelPath  = "../../models/rfdetr-base.onnx";
    std::string videoPath  = "../../data/video.mp4";
    std::string labelsPath = "../../models/coco.names";
    std::string outputDir  = "../../outputs/det/";

    if (argc > 1) modelPath  = argv[1];
    if (argc > 2) videoPath  = argv[2];
    if (argc > 3) labelsPath = argv[3];

    utils::printUsage(argv[0], "Video Detection", modelPath, videoPath, labelsPath);

    try {
        std::cout << "Loading model: " << modelPath << "\n";
        RFDETRDetector detector(modelPath, labelsPath, BackendType::CPU);
        std::cout << "Model loaded.\n";

        cv::VideoCapture cap(videoPath);
        if (!cap.isOpened()) {
            std::cerr << "Cannot open video: " << videoPath << "\n";
            return -1;
        }

        int    frameW      = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
        int    frameH      = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
        double fps         = cap.get(cv::CAP_PROP_FPS);
        int    totalFrames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
        std::cout << "Video: " << frameW << "x" << frameH
                  << " @ " << fps << " FPS, " << totalFrames << " frames\n";

        std::string outPath = utils::getVideoOutputPath(videoPath, outputDir);
        cv::VideoWriter writer(outPath, cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
                               fps, cv::Size(frameW, frameH));
        if (!writer.isOpened()) {
            std::cerr << "Cannot create video writer\n";
            return -1;
        }
        std::cout << "Output: " << outPath << "\n\nProcessing...\n";

        cv::Mat frame;
        int frameCount = 0;
        auto startTime = std::chrono::high_resolution_clock::now();

        while (cap.read(frame)) {
            ++frameCount;
            auto dets = detector.detect(frame);
            detector.drawDetections(frame, dets);

            std::string info = "Frame: " + std::to_string(frameCount) +
                               "/" + std::to_string(totalFrames) +
                               "  Objects: " + std::to_string(dets.size());
            cv::putText(frame, info, {10, 30},
                        cv::FONT_HERSHEY_SIMPLEX, 0.7, {0, 255, 0}, 2);
            writer.write(frame);

            if (frameCount % 30 == 0)
                std::cout << "Processed " << frameCount << "/" << totalFrames << " frames\n";

            cv::imshow("RF-DETR Video Detection", frame);
            if (cv::waitKey(1) == 'q') break;
        }

        auto endTime   = std::chrono::high_resolution_clock::now();
        auto duration  = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
        double avgFps  = duration.count() > 0 ? (frameCount * 1000.0) / duration.count() : 0.0;

        cap.release();
        writer.release();
        cv::destroyAllWindows();

        utils::printMetrics("Video Detection", duration.count(), avgFps);
        std::cout << "Saved: " << outPath << "\n";

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return -1;
    }
    return 0;
}
