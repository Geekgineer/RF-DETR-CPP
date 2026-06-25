// ============================================================================
// RFDETRs-CPP — Example: camera detection
// ============================================================================
// Live camera feed with real-time RF-DETR detection overlay.
//
// Usage: example_camera_det [model.onnx] [labels.names] [camera_id=0]
//
// Controls: 'q' = quit, 's' = save snapshot
// ============================================================================

#include <chrono>
#include <iostream>
#include <string>

#include <opencv2/opencv.hpp>

#include "rfdetrs/rfdetrs.hpp"
#include "utils.hpp"

using namespace rfdetrs;
using namespace rfdetrs::det;

int main(int argc, char* argv[]) {
    std::string modelPath  = "../../models/rfdetr-base.onnx";
    std::string labelsPath = "../../models/coco.names";
    int cameraId = 0;

    if (argc > 1) modelPath  = argv[1];
    if (argc > 2) labelsPath = argv[2];
    if (argc > 3) cameraId  = std::stoi(argv[3]);

    std::cout << "\n+----------------------------------------------------+\n";
    std::cout << "|  RFDETRs-CPP Real-Time Camera Detection\n";
    std::cout << "+----------------------------------------------------+\n";
    std::cout << "Model:     " << modelPath  << "\n";
    std::cout << "Labels:    " << labelsPath << "\n";
    std::cout << "Camera ID: " << cameraId  << "\n";
    std::cout << "\nPress 'q' to quit, 's' to save snapshot\n\n";

    try {
        std::cout << "Loading model...\n";
        RFDETRDetector detector(modelPath, labelsPath, BackendType::CPU);
        std::cout << "Model loaded.\n";

        cv::VideoCapture cap(cameraId);
        if (!cap.isOpened()) {
            std::cerr << "Cannot open camera " << cameraId << "\n";
            return -1;
        }
        std::cout << "Camera opened. Starting detection...\n\n";

        cv::Mat frame;
        int frameCount = 0;
        auto startTime = std::chrono::high_resolution_clock::now();

        while (true) {
            cap >> frame;
            if (frame.empty()) break;
            ++frameCount;

            auto t0   = std::chrono::high_resolution_clock::now();
            auto dets = detector.detect(frame);
            auto ms   = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::high_resolution_clock::now() - t0).count();

            detector.drawDetections(frame, dets);

            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::high_resolution_clock::now() - startTime).count();
            double fps = (elapsed > 0) ? frameCount / static_cast<double>(elapsed) : 0.0;

            std::string info = "FPS: " + std::to_string(static_cast<int>(fps)) +
                               "  Objects: " + std::to_string(dets.size()) +
                               "  Latency: " + std::to_string(ms) + "ms";
            cv::putText(frame, info, {10, 30},
                        cv::FONT_HERSHEY_SIMPLEX, 0.7, {0, 255, 0}, 2);

            cv::imshow("RF-DETR Camera (q=quit, s=snapshot)", frame);
            char key = static_cast<char>(cv::waitKey(1));
            if (key == 'q') break;
            if (key == 's') {
                std::string out = utils::saveImage(frame, "snapshot.jpg", "../../outputs/det/");
                std::cout << "Snapshot saved: " << out << "\n";
            }
        }

        cap.release();
        cv::destroyAllWindows();
        std::cout << "\nStopped. Processed " << frameCount << " frames.\n";

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return -1;
    }
    return 0;
}
