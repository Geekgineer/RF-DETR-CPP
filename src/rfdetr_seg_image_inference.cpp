// SPDX-FileCopyrightText: 2024 Geekgineer
// SPDX-License-Identifier: AGPL-3.0-only
//
// ============================================================================
// rfdetr_seg_image_inference — RF-DETR segmentation CLI
// ============================================================================
// Usage:
//   rfdetr_seg_image_inference <model.onnx> <image_or_dir>
//       [labels.names] [conf=0.5] [top_k=300] [backend=cpu|cuda|trt]
//
// Writes annotated images to <input_dir>/seg_results/
// and a JSON summary to   <input_dir>/seg_results/seg_results.json
// ============================================================================

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "rfdetrs/rfdetrs.hpp"

namespace fs = std::filesystem;
using namespace rfdetrs;
using namespace rfdetrs::seg;

static BackendType parseBackend(const std::string& s) {
    if (s == "cuda") return BackendType::CUDA;
    if (s == "trt" || s == "tensorrt") return BackendType::TensorRT;
    return BackendType::CPU;
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: rfdetr_seg_image_inference <model.onnx> <image_or_dir>"
                     " [labels.names] [conf=0.5] [top_k=300] [backend=cpu|cuda|trt]\n";
        return 1;
    }

    const std::string modelPath  = argv[1];
    const std::string inputPath  = argv[2];
    const std::string labelsPath = (argc > 3) ? argv[3] : "";
    const float       conf       = (argc > 4) ? std::stof(argv[4]) : 0.5f;
    const int         topK       = (argc > 5) ? std::stoi(argv[5]) : 300;
    const BackendType backend    = (argc > 6) ? parseBackend(argv[6]) : BackendType::CPU;

    RFDETRSegDetector detector(modelPath, labelsPath, backend);

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

    // Output directory.
    fs::path outDir = images[0].parent_path() / "seg_results";
    fs::create_directories(outDir);

    std::cout << "RF-DETR Segmentation — model: " << modelPath << "\n"
              << "Processing " << images.size() << " image(s) ...\n";

    // JSON accumulator.
    std::ofstream jsonOut(outDir / "seg_results.json");
    jsonOut << "[\n";

    for (size_t i = 0; i < images.size(); ++i) {
        cv::Mat img = cv::imread(images[i].string());
        if (img.empty()) {
            std::cerr << "[SKIP] Cannot load: " << images[i] << "\n";
            continue;
        }

        auto dets = detector.detect(img, conf, topK);
        detector.drawDetections(img, dets);

        fs::path outImg = outDir / (images[i].stem().string() + "_seg.jpg");
        cv::imwrite(outImg.string(), img);

        const auto& t = detector.getLastTiming();
        std::cout << "  [" << (i + 1) << "/" << images.size() << "] "
                  << images[i].filename().string()
                  << "  detections=" << dets.size()
                  << "  pre=" << static_cast<int>(t.preprocess) << "ms"
                  << "  inf=" << static_cast<int>(t.inference)  << "ms"
                  << "  post=" << static_cast<int>(t.postprocess) << "ms\n";

        // Append JSON entry.
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
                    << ",\"height\":" << d.box.height << "}}";
        }
        jsonOut << "]}";
    }

    jsonOut << "\n]\n";
    std::cout << "\nResults saved to: " << (outDir / "seg_results.json") << "\n";
    return 0;
}
