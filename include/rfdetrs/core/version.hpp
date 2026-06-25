// SPDX-FileCopyrightText: 2024 Geekgineer
// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

// ============================================================================
// RFDETRs-CPP — Library version
// ============================================================================

#define RFDETRS_VERSION_MAJOR 1
#define RFDETRS_VERSION_MINOR 0
#define RFDETRS_VERSION_PATCH 0
#define RFDETRS_VERSION_STRING "1.0.0"

// Minimum supported ONNX Runtime version (must have opset-16 support for
// deformable attention and grid_sample used by RF-DETR exports)
#define RFDETRS_ORT_VERSION_MIN "1.16.0"
