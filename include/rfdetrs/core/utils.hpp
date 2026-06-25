// SPDX-FileCopyrightText: 2024 Geekgineer
// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

// ============================================================================
// RFDETRs-CPP Core Utilities
// ============================================================================

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include <opencv2/opencv.hpp>

namespace rfdetrs {
namespace utils {

inline std::string trimString(const std::string& s) {
    size_t a = 0;
    while (a < s.size() && std::isspace(static_cast<unsigned char>(s[a]))) ++a;
    size_t b = s.size();
    while (b > a && std::isspace(static_cast<unsigned char>(s[b - 1]))) --b;
    return s.substr(a, b - a);
}

template <typename T>
inline typename std::enable_if<std::is_arithmetic<T>::value, T>::type
clamp(const T& value, const T& low, const T& high) {
    T lo = low < high ? low : high;
    T hi = low < high ? high : low;
    if (value < lo) return lo;
    if (value > hi) return hi;
    return value;
}

inline size_t vectorProduct(const std::vector<int64_t>& shape) {
    if (shape.empty()) return 0;
    return std::accumulate(shape.begin(), shape.end(), size_t{1}, std::multiplies<size_t>());
}

inline float sigmoid(float x) noexcept {
    return 1.0f / (1.0f + std::exp(-x));
}

// Inverse sigmoid: logit(p) = ln(p / (1-p)).
// sigmoid(x) >= p  ⟺  x >= logit(p) for p in (0,1).
// Pre-computing the logit threshold lets postprocess skip exp() on most logits.
inline float logit(float p) noexcept {
    return std::log(p / (1.0f - p));
}

inline std::vector<std::string> getClassNames(const std::string& path) {
    std::vector<std::string> names;
    const std::string p = trimString(path);
    if (p.empty()) return names;
    std::ifstream f(p);
    if (!f) {
        std::cerr << "[ERROR] Cannot open labels file: " << p << "\n";
        return names;
    }
    std::string line;
    while (std::getline(f, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (!line.empty()) names.emplace_back(line);
    }
    return names;
}

inline std::vector<cv::Scalar> generateColors(const std::vector<std::string>& classNames, int seed = 42) {
    static std::unordered_map<size_t, std::vector<cv::Scalar>> cache;
    size_t key = 0;
    for (const auto& n : classNames)
        key ^= std::hash<std::string>{}(n) + 0x9e3779b9 + (key << 6) + (key >> 2);
    auto it = cache.find(key);
    if (it != cache.end()) return it->second;

    std::vector<cv::Scalar> colors;
    colors.reserve(classNames.size());
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> dist(0, 255);
    for (size_t i = 0; i < classNames.size(); ++i)
        colors.emplace_back(dist(rng), dist(rng), dist(rng));
    cache[key] = colors;
    return colors;
}

} // namespace utils
} // namespace rfdetrs
