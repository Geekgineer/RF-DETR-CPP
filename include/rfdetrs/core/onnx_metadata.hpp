// SPDX-FileCopyrightText: 2024 Geekgineer
// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

// ============================================================================
// RFDETRs-CPP — ONNX custom metadata reader
// ============================================================================
// Attempts to read a `names` custom metadata entry (Ultralytics-style dict
// string) from an ONNX session.  RF-DETR models do not embed this metadata,
// so this module is a no-op for them — it returns an empty vector and the
// caller falls back to the supplied .names file.

#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <cctype>
#include <regex>
#include <string>
#include <utility>
#include <vector>

namespace rfdetrs {
namespace onnxmeta {

inline std::vector<std::string> parseNamesMetadata(const std::string& raw) {
    std::vector<std::pair<int, std::string>> pairs;
    // Handles both {0: 'name', ...} and {0: "name", ...}
    std::regex re(R"((\d+)\s*:\s*['\"]([^'\"]*)['\"])");
    for (std::sregex_iterator it(raw.begin(), raw.end(), re), end; it != end; ++it) {
        pairs.emplace_back(std::stoi((*it)[1].str()), (*it)[2].str());
    }
    if (pairs.empty()) return {};
    std::sort(pairs.begin(), pairs.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });
    std::vector<std::string> out;
    out.reserve(pairs.size());
    for (const auto& p : pairs) out.push_back(p.second);
    return out;
}

/// Read `names` custom metadata from an ONNX session; returns {} if absent.
inline std::vector<std::string> tryGetEmbeddedClassNames(const Ort::Session& session) {
    try {
        Ort::AllocatorWithDefaultOptions allocator;
        Ort::ModelMetadata meta = session.GetModelMetadata();
        auto namesAlloc = meta.LookupCustomMetadataMapAllocated("names", allocator);
        if (!namesAlloc) return {};
        const char* p = namesAlloc.get();
        if (!p || !*p) return {};
        return parseNamesMetadata(std::string(p));
    } catch (...) {
        return {};
    }
}

} // namespace onnxmeta
} // namespace rfdetrs
