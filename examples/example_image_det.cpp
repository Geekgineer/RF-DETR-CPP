// ============================================================================
// RFDETRs-CPP — Example: image detection
// ============================================================================
// Loads an RF-DETR ONNX model, runs detection on a single image or directory,
// prints results, draws bounding boxes, and saves annotated output.
//
// Usage: example_image_det [model.onnx] [image_or_dir] [labels.names]
// ============================================================================

#include <chrono>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <vector>

#include <opencv2/opencv.hpp>

#include "rfdetrs/rfdetrs.hpp"
#include "utils.hpp"

using namespace rfdetrs;
using namespace rfdetrs::det;
namespace fs = std::filesystem;

int main(int argc, char* argv[]) {
    std::string modelPath  = "../../models/rfdetr-base.onnx";
    std::string inputPath  = "../../data/dog.jpg";
    std::string labelsPath = "../../models/coco.names";
    std::string outputDir  = "../../outputs/det/";

    if (argc > 1 && (std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h")) {
        utils::printUsage(argv[0], "Image Detection", modelPath, inputPath, labelsPath);
        return 0;
    }
    if (argc > 1) modelPath  = argv[1];
    if (argc > 2) inputPath  = argv[2];
    if (argc > 3) labelsPath = argv[3];

    utils::printUsage(argv[0], "Image Detection", modelPath, inputPath, labelsPath);

    // Collect images
    std::vector<std::string> imageFiles;
    try {
        if (fs::is_directory(inputPath)) {
            for (const auto& e : fs::directory_iterator(inputPath))
                if (e.is_regular_file() && utils::isImageFile(e.path().string()))
                    imageFiles.push_back(fs::absolute(e.path()).string());
            if (imageFiles.empty()) {
                std::cerr << "No images found in: " << inputPath << "\n";
                return -1;
            }
        } else if (fs::is_regular_file(inputPath)) {
            imageFiles.push_back(inputPath);
        } else {
            std::cerr << "Invalid path: " << inputPath << "\n";
            return -1;
        }
    } catch (const fs::filesystem_error& e) {
        std::cerr << "Filesystem error: " << e.what() << "\n";
        return -1;
    }

    try {
        std::cout << "Loading model: " << modelPath << "\n";
        RFDETRDetector detector(modelPath, labelsPath, BackendType::CPU);
        std::cout << "Model loaded.\n";

        for (const auto& imgPath : imageFiles) {
            std::cout << "\nProcessing: " << imgPath << "\n";
            cv::Mat image = cv::imread(imgPath);
            if (image.empty()) {
                std::cerr << "Cannot load: " << imgPath << "\n";
                continue;
            }

            auto t0   = std::chrono::high_resolution_clock::now();
            auto dets = detector.detect(image);
            auto ms   = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::high_resolution_clock::now() - t0).count();

            std::cout << "Found " << dets.size() << " objects:\n";
            for (size_t i = 0; i < dets.size(); ++i) {
                std::cout << "  [" << i << "] class=" << dets[i].classId
                          << "  conf=" << std::fixed << std::setprecision(2) << dets[i].conf
                          << "  box=(" << dets[i].box.x << "," << dets[i].box.y
                          << " " << dets[i].box.width << "x" << dets[i].box.height << ")\n";
            }

            cv::Mat result = image.clone();
            detector.drawDetections(result, dets);

            std::string out = utils::saveImage(result, imgPath, outputDir);
            std::cout << "Saved: " << out << "\n";

            utils::printMetrics("Detection", ms);

            cv::imshow("RF-DETR Detection", result);
            std::cout << "Press any key to continue...\n";
            cv::waitKey(0);
        }

        cv::destroyAllWindows();
        std::cout << "\nAll images processed.\n";

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return -1;
    }
    return 0;
}
