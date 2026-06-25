#pragma once
// ============================================================================
// RFDETRs-CPP — Example utilities
// ============================================================================

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

#include <opencv2/opencv.hpp>

namespace utils {

inline std::string getTimestamp() {
    auto now    = std::chrono::system_clock::now();
    auto now_t  = std::chrono::system_clock::to_time_t(now);
    std::tm tm{};
#ifdef _WIN32
    localtime_s(&tm, &now_t);
#else
    localtime_r(&now_t, &tm);
#endif
    std::ostringstream oss;
    oss << std::put_time(&tm, "%Y%m%d_%H%M%S");
    return oss.str();
}

inline std::string getOutputPath(const std::string& inputPath,
                                 const std::string& outputDir,
                                 const std::string& suffix = "_result") {
    namespace fs = std::filesystem;
    fs::create_directories(outputDir);
    fs::path p(inputPath);
    std::string out = p.stem().string() + "_" + getTimestamp() + suffix + p.extension().string();
    return (fs::path(outputDir) / out).string();
}

inline std::string saveImage(const cv::Mat& image,
                             const std::string& inputPath,
                             const std::string& outputDir) {
    std::string out = getOutputPath(inputPath, outputDir);
    cv::imwrite(out, image);
    return out;
}

inline std::string getVideoOutputPath(const std::string& inputPath,
                                      const std::string& outputDir) {
    return getOutputPath(inputPath, outputDir, "_result");
}

inline void printMetrics(const std::string& taskName,
                         int64_t durationMs,
                         double fps = -1) {
    std::cout << "\n==========================================\n";
    std::cout << "  " << taskName << " Metrics\n";
    std::cout << "==========================================\n";
    std::cout << "  Inference Time: " << durationMs << " ms\n";
    if (fps > 0)
        std::cout << "  FPS: " << std::fixed << std::setprecision(2) << fps << "\n";
    std::cout << "==========================================\n\n";
}

inline bool isImageFile(const std::string& path) {
    std::string ext = std::filesystem::path(path).extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    return ext == ".jpg" || ext == ".jpeg" || ext == ".png" ||
           ext == ".bmp" || ext == ".tiff" || ext == ".tif";
}

inline bool isVideoFile(const std::string& path) {
    std::string ext = std::filesystem::path(path).extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    return ext == ".mp4" || ext == ".avi" || ext == ".mov" ||
           ext == ".mkv" || ext == ".flv" || ext == ".wmv";
}

inline void printUsage(const std::string& prog,
                       const std::string& task,
                       const std::string& defModel,
                       const std::string& defInput,
                       const std::string& defLabels) {
    std::cout << "\n+----------------------------------------------------+\n";
    std::cout << "|  RFDETRs-CPP " << task << " Example\n";
    std::cout << "+----------------------------------------------------+\n";
    std::cout << "\nUsage: " << prog << " [model_path] [input_path] [labels_path]\n";
    std::cout << "\nDefaults:\n";
    std::cout << "  model_path  : " << defModel  << "\n";
    std::cout << "  input_path  : " << defInput  << "\n";
    std::cout << "  labels_path : " << defLabels << "\n\n";
}

} // namespace utils
