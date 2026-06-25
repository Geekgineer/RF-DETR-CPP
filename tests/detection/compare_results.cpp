// SPDX-FileCopyrightText: 2024 Geekgineer
// SPDX-License-Identifier: AGPL-3.0-only
//
// ============================================================================
// RFDETRs-CPP — Parity test: compare C++ output against Python ORT reference
// ============================================================================
// Both inference runners use the same ONNX graph and (crucially) the same
// preprocessing (cv2 INTER_LINEAR + ImageNet normalize), so the only source
// of divergence is float32 ordering in the topk.  Tolerances are tight:
//   BBox  : ±2 px
//   Score : ±0.002
//   Count : exact
// ============================================================================

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include <cmath>
#include <fstream>
#include <map>
#include <set>
#include <string>
#include <vector>

using json = nlohmann::json;

// Tight tolerances — same ONNX graph + same ORT, only FP32 ties differ
constexpr double CONF_TOL = 0.002;
constexpr int    BBOX_TOL = 2;

// BASE_PATH_DETECTION is injected by CMakeLists.txt as a quoted string literal
static json readJson(const std::string& path) {
    std::ifstream f(path);
    EXPECT_TRUE(f.is_open()) << "Cannot open: " << path;
    json j;
    f >> j;
    return j;
}

class ParityFixture : public ::testing::Test {
protected:
    json python_results;
    json cpp_results;
    std::string base = BASE_PATH_DETECTION;

    void SetUp() override {
        python_results = readJson(base + "results/results_python.json");
        cpp_results    = readJson(base + "results/results_cpp.json");
    }
};

TEST_F(ParityFixture, ResultsNotEmpty) {
    ASSERT_FALSE(python_results.empty()) << "Python results are empty";
    ASSERT_FALSE(cpp_results.empty())    << "C++ results are empty";
}

TEST_F(ParityFixture, ModelKeysMatch) {
    std::set<std::string> pyKeys, cppKeys;
    for (auto& el : python_results.items()) pyKeys.insert(el.key());
    for (auto& el : cpp_results.items())    cppKeys.insert(el.key());
    for (const auto& k : pyKeys)
        EXPECT_TRUE(cppKeys.count(k)) << "Model '" << k << "' missing in C++ results";
}

TEST_F(ParityFixture, DetectionCountsMatch) {
    for (auto& el : python_results.items()) {
        const std::string& model = el.key();
        if (!cpp_results.contains(model)) continue;

        auto& pyImgs  = el.value()["results"];
        auto& cppImgs = cpp_results[model]["results"];

        for (size_t i = 0; i < pyImgs.size(); ++i) {
            std::string imgPath = pyImgs[i].value("image_path", "");
            size_t pyCount  = pyImgs[i]["inference_results"].size();

            for (size_t j = 0; j < cppImgs.size(); ++j) {
                if (cppImgs[j].value("image_path", "") != imgPath) continue;
                size_t cppCount = cppImgs[j]["inference_results"].size();
                EXPECT_EQ(pyCount, cppCount)
                    << "Detection count mismatch for model=" << model
                    << " image=" << imgPath
                    << " python=" << pyCount << " cpp=" << cppCount;
                break;
            }
        }
    }
}

TEST_F(ParityFixture, DetectionsMatchWithinTolerance) {
    for (auto& el : python_results.items()) {
        const std::string& model = el.key();
        if (!cpp_results.contains(model)) continue;

        auto& pyImgs  = el.value()["results"];
        auto& cppImgs = cpp_results[model]["results"];

        for (size_t i = 0; i < pyImgs.size(); ++i) {
            std::string imgPath = pyImgs[i].value("image_path", "");
            auto& pyDets = pyImgs[i]["inference_results"];

            const json* cppImgPtr = nullptr;
            for (size_t j = 0; j < cppImgs.size(); ++j) {
                if (cppImgs[j].value("image_path", "") == imgPath) {
                    cppImgPtr = &cppImgs[j];
                    break;
                }
            }
            if (!cppImgPtr) continue;
            auto& cppDets = (*cppImgPtr)["inference_results"];

            // For each Python detection, find a matching C++ detection.
            std::vector<bool> matched(cppDets.size(), false);
            for (const auto& pyDet : pyDets) {
                int pyClass   = pyDet.value("class_id", -1);
                double pyConf = pyDet.value("confidence", 0.0);
                int pyL = pyDet["bbox"]["left"];
                int pyT = pyDet["bbox"]["top"];
                int pyW = pyDet["bbox"]["width"];
                int pyH = pyDet["bbox"]["height"];

                bool found = false;
                for (size_t k = 0; k < cppDets.size(); ++k) {
                    if (matched[k]) continue;
                    if (cppDets[k].value("class_id", -2) != pyClass) continue;

                    int cL = cppDets[k]["bbox"]["left"];
                    int cT = cppDets[k]["bbox"]["top"];
                    int cW = cppDets[k]["bbox"]["width"];
                    int cH = cppDets[k]["bbox"]["height"];

                    if (std::abs(pyL - cL) <= BBOX_TOL &&
                        std::abs(pyT - cT) <= BBOX_TOL &&
                        std::abs(pyW - cW) <= BBOX_TOL &&
                        std::abs(pyH - cH) <= BBOX_TOL) {
                        double cConf = cppDets[k].value("confidence", 0.0);
                        EXPECT_NEAR(pyConf, cConf, CONF_TOL)
                            << "Confidence mismatch model=" << model
                            << " image=" << imgPath
                            << " class=" << pyClass;
                        matched[k] = true;
                        found = true;
                        break;
                    }
                }
                EXPECT_TRUE(found)
                    << "No C++ match for Python detection:"
                    << " model=" << model << " image=" << imgPath
                    << " class=" << pyClass
                    << " box=[" << pyL << "," << pyT << "," << pyW << "," << pyH << "]";
            }
        }
    }
}

// Reverse match: every C++ detection must have a corresponding Python detection.
// Catches spurious extra detections produced only by the C++ postprocessor.
TEST_F(ParityFixture, ReverseMatch) {
    for (auto& el : cpp_results.items()) {
        const std::string& model = el.key();
        if (!python_results.contains(model)) continue;

        auto& cppImgs = el.value()["results"];
        auto& pyImgs  = python_results[model]["results"];

        for (size_t i = 0; i < cppImgs.size(); ++i) {
            std::string imgPath = cppImgs[i].value("image_path", "");
            auto& cppDets = cppImgs[i]["inference_results"];

            const json* pyImgPtr = nullptr;
            for (size_t j = 0; j < pyImgs.size(); ++j) {
                if (pyImgs[j].value("image_path", "") == imgPath) {
                    pyImgPtr = &pyImgs[j];
                    break;
                }
            }
            if (!pyImgPtr) continue;
            auto& pyDets = (*pyImgPtr)["inference_results"];

            std::vector<bool> matched(pyDets.size(), false);
            for (const auto& cDet : cppDets) {
                int cClass = cDet.value("class_id", -1);
                int cL = cDet["bbox"]["left"];
                int cT = cDet["bbox"]["top"];
                int cW = cDet["bbox"]["width"];
                int cH = cDet["bbox"]["height"];

                bool found = false;
                for (size_t k = 0; k < pyDets.size(); ++k) {
                    if (matched[k]) continue;
                    if (pyDets[k].value("class_id", -2) != cClass) continue;
                    int pL = pyDets[k]["bbox"]["left"];
                    int pT = pyDets[k]["bbox"]["top"];
                    int pW = pyDets[k]["bbox"]["width"];
                    int pH = pyDets[k]["bbox"]["height"];
                    if (std::abs(cL - pL) <= BBOX_TOL &&
                        std::abs(cT - pT) <= BBOX_TOL &&
                        std::abs(cW - pW) <= BBOX_TOL &&
                        std::abs(cH - pH) <= BBOX_TOL) {
                        matched[k] = true;
                        found = true;
                        break;
                    }
                }
                EXPECT_TRUE(found)
                    << "Spurious C++ detection with no Python counterpart:"
                    << " model=" << model << " image=" << imgPath
                    << " class=" << cClass
                    << " box=[" << cL << "," << cT << "," << cW << "," << cH << "]";
            }
        }
    }
}

// Confidence scores in both outputs must be in (0, 1].
TEST_F(ParityFixture, ConfidenceRange) {
    auto checkConf = [](const json& results, const std::string& side) {
        for (auto& modelEl : results.items()) {
            for (auto& imgEl : modelEl.value()["results"]) {
                for (auto& det : imgEl["inference_results"]) {
                    double conf = det.value("confidence", -1.0);
                    EXPECT_GT(conf, 0.0)
                        << side << " confidence <= 0: model=" << modelEl.key()
                        << " image=" << imgEl.value("image_path", "?");
                    EXPECT_LE(conf, 1.0)
                        << side << " confidence > 1: model=" << modelEl.key()
                        << " image=" << imgEl.value("image_path", "?");
                }
            }
        }
    };
    checkConf(python_results, "Python");
    checkConf(cpp_results,    "C++");
}

// All bounding boxes must have positive width and height, and non-negative origin.
TEST_F(ParityFixture, BBoxSanity) {
    auto checkBoxes = [](const json& results, const std::string& side) {
        for (auto& modelEl : results.items()) {
            for (auto& imgEl : modelEl.value()["results"]) {
                std::string img = imgEl.value("image_path", "?");
                for (auto& det : imgEl["inference_results"]) {
                    int x = det["bbox"].value("left",   -1);
                    int y = det["bbox"].value("top",    -1);
                    int w = det["bbox"].value("width",  -1);
                    int h = det["bbox"].value("height", -1);
                    EXPECT_GE(x, 0) << side << " bbox.left < 0, image=" << img;
                    EXPECT_GE(y, 0) << side << " bbox.top  < 0, image=" << img;
                    EXPECT_GT(w, 0) << side << " bbox.width <= 0, image=" << img;
                    EXPECT_GT(h, 0) << side << " bbox.height <= 0, image=" << img;
                }
            }
        }
    };
    checkBoxes(python_results, "Python");
    checkBoxes(cpp_results,    "C++");
}

// Detections within each result set must be sorted in non-increasing confidence order.
TEST_F(ParityFixture, ScoreOrdering) {
    auto checkOrder = [](const json& results, const std::string& side) {
        for (auto& modelEl : results.items()) {
            for (auto& imgEl : modelEl.value()["results"]) {
                std::string img = imgEl.value("image_path", "?");
                auto& dets = imgEl["inference_results"];
                for (size_t i = 1; i < dets.size(); ++i) {
                    double prev = dets[i - 1].value("confidence", 0.0);
                    double curr = dets[i    ].value("confidence", 0.0);
                    EXPECT_GE(prev, curr - CONF_TOL)
                        << side << " detections not sorted descending at index "
                        << i << ", image=" << img
                        << " prev=" << prev << " curr=" << curr;
                }
            }
        }
    };
    checkOrder(python_results, "Python");
    checkOrder(cpp_results,    "C++");
}
