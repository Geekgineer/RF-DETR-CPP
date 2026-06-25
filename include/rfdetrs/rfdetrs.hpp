// SPDX-FileCopyrightText: 2024 Geekgineer
// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

// ============================================================================
// RFDETRs-CPP — Master Include
// ============================================================================
// Single-header entry point.  Include this file to access the full API.
//
//   #include "rfdetrs/rfdetrs.hpp"
//   using rfdetrs::BackendType;
//   using rfdetrs::det::RFDETRDetector;
//   using rfdetrs::det::Detection;
// ============================================================================

#include "rfdetrs/core/version.hpp"
#include "rfdetrs/core/session_base.hpp"
#include "rfdetrs/tasks/detection.hpp"
#include "rfdetrs/tasks/segmentation.hpp"
#include "rfdetrs/tasks/keypoint.hpp"
