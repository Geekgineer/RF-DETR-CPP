// SPDX-FileCopyrightText: 2024 Geekgineer
// SPDX-License-Identifier: AGPL-3.0-only
//
// ============================================================================
// RFDETRs-CPP — C++ inference test runner
// ============================================================================
// Generates tests/detection/results/results_cpp.json for parity comparison.
//
// Usage (from build directory):
//   ./inference_rfdetr_cpp <model_dir> <images_dir> [conf=0.5] [top_k=300]
//
// Defaults load from tests/detection/  (relative to executable's parent).
// ============================================================================

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>
#include <opencv2/imgcodecs.hpp>

#include "rfdetrs/rfdetrs.hpp"

namespace fs = std::filesystem;
using json = nlohmann::json;
using namespace rfdetrs;
using namespace rfdetrs::det;

static std::vector<fs::path> collectImages(const fs::path& dir) {
    static const std::vector<std::string> exts = {".jpg", ".jpeg", ".png", ".bmp"};
    std::vector<fs::path> files;
    for (const auto& e : fs::directory_iterator(dir)) {
        if (!e.is_regular_file()) continue;
        std::string ext = e.path().extension().string();
        for (auto& c : ext) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        if (std::find(exts.begin(), exts.end(), ext) != exts.end())
            files.push_back(e.path());
    }
    std::sort(files.begin(), files.end());
    return files;
}

int main(int argc, char* argv[]) {
    // Walk up from the binary location until we find tests/detection/,
    // supporting both build/inference_rfdetr_cpp and build/tests/inference_rfdetr_cpp.
    auto findTestDir = [](fs::path start) -> fs::path {
        for (fs::path p = start; p != p.parent_path(); p = p.parent_path()) {
            fs::path candidate = p / "tests" / "detection";
            if (fs::is_directory(candidate)) return candidate;
        }
        return "";
    };
    fs::path testDir = findTestDir(fs::path(argv[0]).parent_path());
    if (testDir.empty()) testDir = fs::current_path() / "tests" / "detection";

    fs::path modelDir  = (argc > 1) ? fs::path(argv[1]) : (testDir / "models");
    fs::path imagesDir = (argc > 2) ? fs::path(argv[2]) : (testDir / "data" / "images");
    float conf    = (argc > 3) ? std::stof(argv[3]) : 0.5f;
    int topK      = (argc > 4) ? std::stoi(argv[4]) : 300;

    // outDir always lives inside tests/detection/results/ (where Python put its json)
    fs::path outDir = testDir / "results";
    fs::create_directories(outDir);

    // Discover .onnx models
    std::vector<fs::path> models;
    for (const auto& e : fs::directory_iterator(modelDir)) {
        if (e.is_regular_file() && e.path().extension() == ".onnx")
            models.push_back(e.path());
    }
    std::sort(models.begin(), models.end());

    if (models.empty()) {
        std::cerr << "[ERROR] No .onnx models in " << modelDir << "\n";
        return 1;
    }

    std::vector<fs::path> images = collectImages(imagesDir);
    if (images.empty()) {
        std::cerr << "[ERROR] No images in " << imagesDir << "\n";
        return 1;
    }

    json allResults;

    for (const auto& modelPath : models) {
        std::cout << "\n[MODEL] " << modelPath.filename() << "\n";
        RFDETRDetector detector(modelPath.string(), "", BackendType::CPU);

        json modelResults;
        modelResults["weights_path"] = modelPath.string();
        modelResults["task"] = "detect";
        modelResults["results"] = json::array();

        for (const auto& imgPath : images) {
            cv::Mat img = cv::imread(imgPath.string());
            if (img.empty()) {
                std::cerr << "[SKIP] " << imgPath << "\n";
                continue;
            }

            auto t0 = std::chrono::high_resolution_clock::now();
            auto dets = detector.detect(img, conf, topK);
            auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::high_resolution_clock::now() - t0).count();

            std::cout << "  " << imgPath.filename().string()
                      << ": " << dets.size() << " dets  " << ms << " ms\n";

            json imgResult;
            imgResult["image_path"] = imgPath.string();
            imgResult["inference_results"] = json::array();

            for (const auto& d : dets) {
                json det;
                det["class_id"]   = d.classId;
                det["confidence"] = static_cast<double>(d.conf);
                det["bbox"] = {
                    {"left",   d.box.x},
                    {"top",    d.box.y},
                    {"width",  d.box.width},
                    {"height", d.box.height},
                };
                imgResult["inference_results"].push_back(det);
            }
            modelResults["results"].push_back(imgResult);
        }

        std::string modelName = modelPath.stem().string();
        allResults[modelName] = modelResults;
    }

    fs::path outPath = outDir / "results_cpp.json";
    std::ofstream f(outPath);
    f << std::setw(2) << allResults << "\n";
    std::cout << "\nSaved: " << outPath << "\n";
    return 0;
}
