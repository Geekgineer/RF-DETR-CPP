// SPDX-FileCopyrightText: 2024 Geekgineer
// SPDX-License-Identifier: AGPL-3.0-only
//
// ============================================================================
// RFDETRs-CPP — Image inference CLI
// ============================================================================
// Usage:
//   rfdetr_image_inference <model.onnx> <image_or_dir> [labels.names]
//                          [conf=0.5] [top_k=300] [backend=cpu|cuda|trt]
//
// Outputs:
//   - Annotated image saved next to input as <stem>_rfdetr.<ext>
//   - JSON detections printed to stdout (and saved as <stem>_rfdetr.json)
// ============================================================================

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgcodecs.hpp>

#include "rfdetrs/rfdetrs.hpp"

namespace fs = std::filesystem;
using namespace rfdetrs;
using namespace rfdetrs::det;

// ---------------------------------------------------------------------------
// JSON helpers (no external dependency)
// ---------------------------------------------------------------------------

static std::string escapeJson(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    for (char c : s) {
        if (c == '"') out += "\\\"";
        else if (c == '\\') out += "\\\\";
        else out += c;
    }
    return out;
}

static std::string detectionsToJson(
    const std::string& modelPath,
    const std::string& imagePath,
    const std::vector<Detection>& dets)
{
    std::ostringstream j;
    j << std::fixed << std::setprecision(6);
    j << "{\n"
      << "  \"model\": \"" << escapeJson(modelPath) << "\",\n"
      << "  \"results\": [\n"
      << "    {\n"
      << "      \"image_path\": \"" << escapeJson(imagePath) << "\",\n"
      << "      \"inference_results\": [\n";

    for (size_t i = 0; i < dets.size(); ++i) {
        const auto& d = dets[i];
        j << "        {\n"
          << "          \"class_id\": " << d.classId << ",\n"
          << "          \"confidence\": " << d.conf << ",\n"
          << "          \"bbox\": {\n"
          << "            \"left\": "   << d.box.x      << ",\n"
          << "            \"top\": "    << d.box.y      << ",\n"
          << "            \"width\": "  << d.box.width  << ",\n"
          << "            \"height\": " << d.box.height << "\n"
          << "          }\n"
          << "        }";
        if (i + 1 < dets.size()) j << ",";
        j << "\n";
    }

    j << "      ]\n"
      << "    }\n"
      << "  ]\n"
      << "}\n";
    return j.str();
}

// ---------------------------------------------------------------------------
// Argument helpers
// ---------------------------------------------------------------------------

static BackendType parseBackend(const std::string& s) {
    if (s == "cuda") return BackendType::CUDA;
    if (s == "trt" || s == "tensorrt") return BackendType::TensorRT;
    return BackendType::CPU;
}

static std::vector<fs::path> collectImages(const fs::path& input) {
    static const std::vector<std::string> exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"};
    std::vector<fs::path> files;
    auto lcExt = [](const fs::path& p) {
        std::string e = p.extension().string();
        for (auto& c : e) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        return e;
    };
    if (fs::is_directory(input)) {
        for (const auto& entry : fs::directory_iterator(input)) {
            if (entry.is_regular_file()) {
                const std::string e = lcExt(entry.path());
                if (std::find(exts.begin(), exts.end(), e) != exts.end())
                    files.push_back(entry.path());
            }
        }
    } else {
        files.push_back(input);
    }
    std::sort(files.begin(), files.end());
    return files;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cout << "Usage: " << argv[0]
                  << " <model.onnx> <image_or_dir> [labels.names]"
                     " [conf=0.5] [top_k=300] [backend=cpu|cuda|trt]\n";
        return 1;
    }

    const std::string modelPath  = argv[1];
    const std::string inputPath  = argv[2];
    const std::string labelsPath = (argc > 3) ? argv[3] : "";
    const float confThreshold    = (argc > 4) ? std::stof(argv[4]) : 0.5f;
    const int topK               = (argc > 5) ? std::stoi(argv[5]) : 300;
    BackendType backend = BackendType::CPU;
    if (argc > 6) backend = parseBackend(argv[6]);

    // Build detector
    RFDETRDetector detector(modelPath, labelsPath, backend);

    // Collect images
    const std::vector<fs::path> images = collectImages(inputPath);
    if (images.empty()) {
        std::cerr << "[ERROR] No images found at: " << inputPath << "\n";
        return 1;
    }

    for (const auto& imgPath : images) {
        std::cout << "\nProcessing: " << imgPath << "\n";

        cv::Mat image = cv::imread(imgPath.string());
        if (image.empty()) {
            std::cerr << "[ERROR] Cannot load image: " << imgPath << "\n";
            continue;
        }

        auto t0 = std::chrono::high_resolution_clock::now();
        std::vector<Detection> dets = detector.detect(image, confThreshold, topK);
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::high_resolution_clock::now() - t0).count();

        std::cout << "Detections: " << dets.size() << "  |  " << ms << " ms\n";
        for (const auto& d : dets) {
            std::cout << "  class=" << d.classId
                      << " conf=" << std::fixed << std::setprecision(3) << d.conf
                      << " box=[" << d.box.x << "," << d.box.y
                      << "," << d.box.width << "," << d.box.height << "]\n";
        }

        // Annotated image
        detector.drawDetections(image, dets);
        fs::path outImg = imgPath.parent_path()
            / (imgPath.stem().string() + "_rfdetr" + imgPath.extension().string());
        cv::imwrite(outImg.string(), image);
        std::cout << "Annotated image: " << outImg << "\n";

        // JSON
        std::string json = detectionsToJson(modelPath, imgPath.string(), dets);
        fs::path outJson = imgPath.parent_path()
            / (imgPath.stem().string() + "_rfdetr.json");
        {
            std::ofstream f(outJson);
            f << json;
        }
        std::cout << "JSON saved:      " << outJson << "\n";
        std::cout << json;
    }

    return 0;
}
