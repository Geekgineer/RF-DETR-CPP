// SPDX-FileCopyrightText: 2024 Geekgineer
// SPDX-License-Identifier: AGPL-3.0-only
//
// ============================================================================
// rfdetr_keypoint_image_inference — RF-DETR keypoint detection CLI
// ============================================================================
// Usage:
//   rfdetr_keypoint_image_inference <model.onnx> <image_or_dir>
//       [labels.names] [conf=0.5] [top_k=300] [backend=cpu|cuda|trt]
//       [kps_per_class=0,17]    <- comma-separated per-class keypoint counts
//
// Example (COCO KeypointPreview model, person=17 kps, background=0):
//   rfdetr_keypoint_image_inference rfdetr-keypoint-preview.onnx dog.jpg
//       "" 0.5 300 cpu 0,17
//
// Writes annotated images to <input_dir>/kp_results/
// ============================================================================

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "rfdetrs/rfdetrs.hpp"

namespace fs = std::filesystem;
using namespace rfdetrs;
using namespace rfdetrs::kp;

static BackendType parseBackend(const std::string& s) {
    if (s == "cuda") return BackendType::CUDA;
    if (s == "trt" || s == "tensorrt") return BackendType::TensorRT;
    return BackendType::CPU;
}

static std::vector<int> parseIntList(const std::string& s) {
    std::vector<int> out;
    std::stringstream ss(s);
    std::string tok;
    while (std::getline(ss, tok, ','))
        out.push_back(std::stoi(tok));
    return out;
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: rfdetr_keypoint_image_inference <model.onnx> <image_or_dir>"
                     " [labels.names] [conf=0.5] [top_k=300] [backend=cpu|cuda|trt]"
                     " [kps_per_class=0,17]\n";
        return 1;
    }

    const std::string modelPath     = argv[1];
    const std::string inputPath     = argv[2];
    const std::string labelsPath    = (argc > 3) ? argv[3] : "";
    const float       conf          = (argc > 4) ? std::stof(argv[4]) : 0.5f;
    const int         topK          = (argc > 5) ? std::stoi(argv[5]) : 300;
    const BackendType backend       = (argc > 6) ? parseBackend(argv[6]) : BackendType::CPU;
    std::vector<int>  kpsPerClass   = (argc > 7) ? parseIntList(argv[7]) : std::vector<int>{};

    RFDETRKeypointDetector detector(modelPath, labelsPath, kpsPerClass, backend);

    std::cout << "Keypoints per class: ";
    for (int n : detector.getNumKeypointsPerClass()) std::cout << n << " ";
    std::cout << "\n";

    // Collect image files.
    std::vector<fs::path> images;
    if (fs::is_directory(inputPath)) {
        for (const auto& entry : fs::directory_iterator(inputPath)) {
            const auto ext = entry.path().extension().string();
            if (ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp")
                images.push_back(entry.path());
        }
        std::sort(images.begin(), images.end());
    } else {
        images.push_back(inputPath);
    }

    if (images.empty()) {
        std::cerr << "[ERROR] No images found at: " << inputPath << "\n";
        return 1;
    }

    fs::path outDir = images[0].parent_path() / "kp_results";
    fs::create_directories(outDir);

    std::cout << "RF-DETR Keypoint — model: " << modelPath << "\n"
              << "Processing " << images.size() << " image(s) ...\n";

    std::ofstream jsonOut(outDir / "kp_results.json");
    jsonOut << "[\n";

    for (size_t i = 0; i < images.size(); ++i) {
        cv::Mat img = cv::imread(images[i].string());
        if (img.empty()) {
            std::cerr << "[SKIP] Cannot load: " << images[i] << "\n";
            continue;
        }

        auto dets = detector.detect(img, conf, topK);
        detector.drawDetections(img, dets);

        fs::path outImg = outDir / (images[i].stem().string() + "_kp.jpg");
        cv::imwrite(outImg.string(), img);

        const auto& t = detector.getLastTiming();
        std::cout << "  [" << (i + 1) << "/" << images.size() << "] "
                  << images[i].filename().string()
                  << "  detections=" << dets.size()
                  << "  pre=" << static_cast<int>(t.preprocess) << "ms"
                  << "  inf=" << static_cast<int>(t.inference)  << "ms"
                  << "  post=" << static_cast<int>(t.postprocess) << "ms\n";

        if (i > 0) jsonOut << ",\n";
        jsonOut << "  {\"image\":\"" << images[i].filename().string()
                << "\",\"detections\":" << dets.size() << ","
                << "\"pre_ms\":" << t.preprocess << ","
                << "\"inf_ms\":" << t.inference  << ","
                << "\"post_ms\":" << t.postprocess
                << ",\"results\":[";
        for (size_t j = 0; j < dets.size(); ++j) {
            if (j > 0) jsonOut << ",";
            const auto& d = dets[j];
            jsonOut << "{\"class_id\":" << d.classId
                    << ",\"confidence\":" << d.conf
                    << ",\"bbox\":{\"left\":" << d.box.x
                    << ",\"top\":"   << d.box.y
                    << ",\"width\":" << d.box.width
                    << ",\"height\":" << d.box.height << "}"
                    << ",\"keypoints\":[";
            for (size_t ki = 0; ki < d.keypoints.size(); ++ki) {
                if (ki > 0) jsonOut << ",";
                const auto& kp = d.keypoints[ki];
                jsonOut << "{\"x\":" << kp.x
                        << ",\"y\":" << kp.y
                        << ",\"confidence\":" << kp.confidence
                        << ",\"visible\":" << kp.visible << "}";
            }
            jsonOut << "]}";
        }
        jsonOut << "]}";
    }

    jsonOut << "\n]\n";
    std::cout << "\nResults saved to: " << (outDir / "kp_results.json") << "\n";
    return 0;
}
