// SPDX-FileCopyrightText: 2024 Geekgineer
// SPDX-License-Identifier: AGPL-3.0-only
//
// ============================================================================
// RFDETRs-CPP — Batch image inference
// ============================================================================
// Processes all images in a directory (or a single file), writes annotated
// output images alongside a JSON summary of all detections.
//
// Usage:
//   rfdetr_batch_inference <model.onnx> <image_or_dir>
//                          [labels.names] [conf=0.5] [top_k=300]
//                          [backend=cpu|cuda|trt]
// ============================================================================

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <opencv2/imgcodecs.hpp>

#include "rfdetrs/rfdetrs.hpp"

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

static std::vector<fs::path> collectImages(const fs::path& p) {
    static const std::vector<std::string> exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"};
    std::vector<fs::path> files;
    auto isImg = [&](const fs::path& f) {
        std::string ext = f.extension().string();
        for (auto& c : ext) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        return std::find(exts.begin(), exts.end(), ext) != exts.end();
    };
    if (fs::is_directory(p)) {
        for (const auto& e : fs::directory_iterator(p))
            if (e.is_regular_file() && isImg(e.path()))
                files.push_back(fs::absolute(e.path()));
    } else if (fs::is_regular_file(p) && isImg(p)) {
        files.push_back(fs::absolute(p));
    }
    std::sort(files.begin(), files.end());
    return files;
}

// Minimal JSON escape
static std::string jsonStr(const std::string& s) {
    std::ostringstream o;
    o << '"';
    for (char c : s) {
        if (c == '"') o << "\\\"";
        else if (c == '\\') o << "\\\\";
        else o << c;
    }
    o << '"';
    return o.str();
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0]
                  << " <model.onnx> <image_or_dir>"
                  << " [labels.names] [conf=0.5] [top_k=300] [backend=cpu|cuda|trt]\n";
        return 1;
    }

    std::string modelPath  = argv[1];
    std::string inputPath  = argv[2];
    std::string labelsPath = (argc > 3) ? argv[3] : "";
    float conf             = (argc > 4) ? std::stof(argv[4]) : 0.5f;
    int   topK             = (argc > 5) ? std::stoi(argv[5]) : 300;
    BackendType backend    = (argc > 6) ? parseBackend(argv[6]) : BackendType::CPU;

    auto images = collectImages(inputPath);
    if (images.empty()) {
        std::cerr << "[ERROR] No images found in: " << inputPath << "\n";
        return 1;
    }

    std::cout << "[INFO] Model:  " << modelPath << "\n";
    std::cout << "[INFO] Images: " << images.size() << "\n";
    std::cout << "[INFO] Conf:   " << conf << "  TopK: " << topK << "\n";

    RFDETRDetector detector(modelPath, labelsPath, backend);

    // Prepare output dir (next to first image's parent)
    fs::path outDir = fs::path(inputPath);
    if (fs::is_regular_file(outDir)) outDir = outDir.parent_path();
    outDir = outDir / "batch_results";
    fs::create_directories(outDir);

    std::ofstream jsonOut(outDir / "batch_results.json");
    jsonOut << "{\n  \"model\": " << jsonStr(modelPath) << ",\n  \"results\": [\n";

    auto totalStart = std::chrono::high_resolution_clock::now();
    size_t totalDets = 0;

    for (size_t i = 0; i < images.size(); ++i) {
        const auto& imgPath = images[i];
        cv::Mat img = cv::imread(imgPath.string());
        if (img.empty()) {
            std::cerr << "[SKIP] Cannot load: " << imgPath << "\n";
            continue;
        }

        auto t0   = std::chrono::high_resolution_clock::now();
        auto dets = detector.detect(img, conf, topK);
        auto ms   = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::high_resolution_clock::now() - t0).count();

        totalDets += dets.size();
        std::cout << "[" << (i + 1) << "/" << images.size() << "] "
                  << imgPath.filename().string()
                  << "  dets=" << dets.size() << "  " << ms << " ms\n";

        // Draw and save annotated image
        detector.drawDetections(img, dets);
        fs::path outImg = outDir / (imgPath.stem().string() + "_rfdetr" + imgPath.extension().string());
        cv::imwrite(outImg.string(), img);

        // JSON entry
        if (i > 0) jsonOut << ",\n";
        jsonOut << "    {\n";
        jsonOut << "      \"image_path\": " << jsonStr(imgPath.string()) << ",\n";
        jsonOut << "      \"inference_ms\": " << ms << ",\n";
        jsonOut << "      \"inference_results\": [\n";
        for (size_t j = 0; j < dets.size(); ++j) {
            const auto& d = dets[j];
            if (j > 0) jsonOut << ",\n";
            jsonOut << "        {\"class_id\": " << d.classId
                    << ", \"confidence\": " << std::fixed << std::setprecision(4) << d.conf
                    << ", \"bbox\": {\"left\": " << d.box.x
                    << ", \"top\": "    << d.box.y
                    << ", \"width\": "  << d.box.width
                    << ", \"height\": " << d.box.height << "}}";
        }
        jsonOut << "\n      ]\n    }";
    }

    auto totalMs = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::high_resolution_clock::now() - totalStart).count();

    jsonOut << "\n  ]\n}\n";
    jsonOut.close();

    std::cout << "\n[SUMMARY] " << images.size() << " images  "
              << totalDets << " detections  "
              << totalMs << " ms total  "
              << std::fixed << std::setprecision(1)
              << (images.size() * 1000.0 / static_cast<double>(totalMs > 0 ? totalMs : 1)) << " img/s\n";
    std::cout << "[INFO] Results saved to: " << outDir << "\n";
    return 0;
}
