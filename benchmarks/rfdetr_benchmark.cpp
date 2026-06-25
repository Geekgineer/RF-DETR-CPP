// ============================================================================
// RFDETRs-CPP — Performance Benchmark
// ============================================================================
// Measures RF-DETR inference latency and throughput across warmup and timed
// iterations.  Reports min / max / avg / median latency and FPS.
//
// Usage:
//   rfdetr_benchmark <model.onnx> [image_or_video] [labels.names]
//                   [--backend cpu|cuda|trt]
//                   [--warmup 10] [--iters 100] [--conf 0.5] [--topk 300]
//                   [--mode image|video|batch]
// ============================================================================

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>

#include "rfdetrs/rfdetrs.hpp"

using namespace rfdetrs;
using namespace rfdetrs::det;
namespace fs = std::filesystem;
using clock_t2 = std::chrono::high_resolution_clock;

// ---------------------------------------------------------------------------
// Arg parsing helpers
// ---------------------------------------------------------------------------
static std::string getArg(int argc, char* argv[], const std::string& flag,
                           const std::string& def = "") {
    for (int i = 1; i < argc - 1; ++i)
        if (std::string(argv[i]) == flag)
            return argv[i + 1];
    return def;
}

static bool hasFlag(int argc, char* argv[], const std::string& flag) {
    for (int i = 1; i < argc; ++i)
        if (std::string(argv[i]) == flag) return true;
    return false;
}

static BackendType parseBackend(const std::string& s) {
    if (s == "cuda") return BackendType::CUDA;
    if (s == "trt")  return BackendType::TensorRT;
    return BackendType::CPU;
}

// ---------------------------------------------------------------------------
// Stats
// ---------------------------------------------------------------------------
struct Stats {
    double minMs, maxMs, avgMs, medianMs, fps;
    size_t count;
};

static Stats computeStats(std::vector<double>& samples, double totalMs) {
    std::sort(samples.begin(), samples.end());
    Stats s;
    s.count    = samples.size();
    s.minMs    = samples.front();
    s.maxMs    = samples.back();
    s.avgMs    = totalMs / static_cast<double>(samples.size());
    s.medianMs = samples[samples.size() / 2];
    s.fps      = (s.avgMs > 0) ? 1000.0 / s.avgMs : 0.0;
    return s;
}

static void printStats(const std::string& label, const Stats& s,
                       const RFDETRDetector::TimingMs* phase = nullptr) {
    std::cout << "\n============================================\n";
    std::cout << "  " << label << "\n";
    std::cout << "============================================\n";
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "  Iterations : " << s.count    << "\n";
    std::cout << "  Min        : " << s.minMs    << " ms\n";
    std::cout << "  Max        : " << s.maxMs    << " ms\n";
    std::cout << "  Avg        : " << s.avgMs    << " ms\n";
    std::cout << "  Median     : " << s.medianMs << " ms\n";
    std::cout << "  FPS        : " << s.fps      << "\n";
    if (phase) {
        std::cout << "--------------------------------------------\n";
        std::cout << "  Phase breakdown (last iter):\n";
        std::cout << "    Preprocess  : " << phase->preprocess  << " ms\n";
        std::cout << "    Inference   : " << phase->inference   << " ms\n";
        std::cout << "    Postprocess : " << phase->postprocess << " ms\n";
    }
    std::cout << "============================================\n";
}

// ---------------------------------------------------------------------------
// Image benchmark
// ---------------------------------------------------------------------------
static void benchmarkImage(RFDETRDetector& detector,
                           const std::string& imagePath,
                           float conf, int topK,
                           int warmup, int iters) {
    cv::Mat image = cv::imread(imagePath);
    if (image.empty()) {
        std::cerr << "[ERROR] Cannot load image: " << imagePath << "\n";
        return;
    }
    std::cout << "\n[IMAGE BENCHMARK] " << imagePath
              << " (" << image.cols << "x" << image.rows << ")\n";
    std::cout << "Warmup " << warmup << " runs...\n";
    for (int i = 0; i < warmup; ++i)
        detector.detect(image, conf, topK);

    std::cout << "Timing " << iters << " runs...\n";
    std::vector<double> latencies;
    latencies.reserve(iters);
    double total = 0.0;
    for (int i = 0; i < iters; ++i) {
        auto t0 = clock_t2::now();
        detector.detect(image, conf, topK);
        double ms = std::chrono::duration<double, std::milli>(clock_t2::now() - t0).count();
        latencies.push_back(ms);
        total += ms;
    }
    auto phase = detector.getLastTiming();
    printStats("Image Inference", computeStats(latencies, total), &phase);
}

// ---------------------------------------------------------------------------
// Video benchmark (max `iters` frames)
// ---------------------------------------------------------------------------
static void benchmarkVideo(RFDETRDetector& detector,
                           const std::string& videoPath,
                           float conf, int topK,
                           int warmup, int iters) {
    cv::VideoCapture cap(videoPath);
    if (!cap.isOpened()) {
        std::cerr << "[ERROR] Cannot open video: " << videoPath << "\n";
        return;
    }
    int W = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int H = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    std::cout << "\n[VIDEO BENCHMARK] " << videoPath << " (" << W << "x" << H << ")\n";

    cv::Mat frame;
    std::cout << "Warmup " << warmup << " frames...\n";
    for (int i = 0; i < warmup && cap.read(frame); ++i)
        detector.detect(frame, conf, topK);

    std::cout << "Timing " << iters << " frames...\n";
    std::vector<double> latencies;
    latencies.reserve(iters);
    double total = 0.0;
    while (static_cast<int>(latencies.size()) < iters && cap.read(frame)) {
        auto t0 = clock_t2::now();
        detector.detect(frame, conf, topK);
        double ms = std::chrono::duration<double, std::milli>(clock_t2::now() - t0).count();
        latencies.push_back(ms);
        total += ms;
    }
    cap.release();
    if (latencies.empty()) {
        std::cerr << "[ERROR] No frames processed.\n";
        return;
    }
    auto phase = detector.getLastTiming();
    printStats("Video Inference", computeStats(latencies, total), &phase);
}

// ---------------------------------------------------------------------------
// Batch benchmark (directory)
// ---------------------------------------------------------------------------
static void benchmarkBatch(RFDETRDetector& detector,
                            const std::string& dirPath,
                            float conf, int topK,
                            int warmup, int maxIters) {
    std::vector<fs::path> images;
    for (const auto& e : fs::directory_iterator(dirPath)) {
        if (!e.is_regular_file()) continue;
        std::string ext = e.path().extension().string();
        for (auto& c : ext) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        if (ext == ".jpg" || ext == ".jpeg" || ext == ".png")
            images.push_back(e.path());
    }
    if (images.empty()) {
        std::cerr << "[ERROR] No images in: " << dirPath << "\n";
        return;
    }
    std::cout << "\n[BATCH BENCHMARK] " << dirPath
              << " (" << images.size() << " images)\n";

    // Warmup
    std::cout << "Warmup " << warmup << " runs...\n";
    for (int i = 0; i < warmup; ++i) {
        cv::Mat img = cv::imread(images[i % images.size()].string());
        if (!img.empty()) detector.detect(img, conf, topK);
    }

    // Timed
    int iters = std::min(maxIters, static_cast<int>(images.size()));
    std::cout << "Timing " << iters << " images...\n";
    std::vector<double> latencies;
    latencies.reserve(iters);
    double total = 0.0;
    for (int i = 0; i < iters; ++i) {
        cv::Mat img = cv::imread(images[i].string());
        if (img.empty()) continue;
        auto t0 = clock_t2::now();
        detector.detect(img, conf, topK);
        double ms = std::chrono::duration<double, std::milli>(clock_t2::now() - t0).count();
        latencies.push_back(ms);
        total += ms;
    }
    auto phase = detector.getLastTiming();
    printStats("Batch Inference", computeStats(latencies, total), &phase);
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main(int argc, char* argv[]) {
    if (argc < 2 || hasFlag(argc, argv, "--help")) {
        std::cout << "Usage: rfdetr_benchmark <model.onnx> [input_path] [labels.names]\n"
                  << "  --backend cpu|cuda|trt  (default: cpu)\n"
                  << "  --mode image|video|batch (default: image)\n"
                  << "  --warmup N  (default: 10)\n"
                  << "  --iters  N  (default: 100)\n"
                  << "  --conf   F  (default: 0.5)\n"
                  << "  --topk   N  (default: 300)\n";
        return 0;
    }

    std::string modelPath  = argv[1];
    std::string inputPath  = (argc > 2 && argv[2][0] != '-') ? argv[2] : "../data/dog.jpg";
    std::string labelsPath = (argc > 3 && argv[3][0] != '-') ? argv[3] : "";
    std::string backendStr = getArg(argc, argv, "--backend", "cpu");
    std::string mode       = getArg(argc, argv, "--mode",    "image");
    int warmup             = std::stoi(getArg(argc, argv, "--warmup", "10"));
    int iters              = std::stoi(getArg(argc, argv, "--iters",  "100"));
    float conf             = std::stof(getArg(argc, argv, "--conf",   "0.5"));
    int topK               = std::stoi(getArg(argc, argv, "--topk",   "300"));

    BackendType backend = parseBackend(backendStr);

    std::cout << "============================================\n";
    std::cout << "  RFDETRs-CPP Benchmark\n";
    std::cout << "============================================\n";
    std::cout << "  Model   : " << modelPath  << "\n";
    std::cout << "  Input   : " << inputPath  << "\n";
    std::cout << "  Backend : " << backendStr << "\n";
    std::cout << "  Mode    : " << mode       << "\n";
    std::cout << "  Warmup  : " << warmup     << "\n";
    std::cout << "  Iters   : " << iters      << "\n";
    std::cout << "  Conf    : " << conf       << "\n";
    std::cout << "  TopK    : " << topK       << "\n";

    try {
        RFDETRDetector detector(modelPath, labelsPath, backend);

        if (mode == "video")
            benchmarkVideo(detector, inputPath, conf, topK, warmup, iters);
        else if (mode == "batch")
            benchmarkBatch(detector, inputPath, conf, topK, warmup, iters);
        else
            benchmarkImage(detector, inputPath, conf, topK, warmup, iters);

    } catch (const std::exception& e) {
        std::cerr << "[ERROR] " << e.what() << "\n";
        return 1;
    }
    return 0;
}
