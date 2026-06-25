#!/bin/bash
# ============================================================================
# RFDETRs-CPP Keypoint Structural Test Runner
# ============================================================================
# Steps:
#   1. Install rfdetr[onnx] (via uv)
#   2. Export RFDETRKeypointPreview (random weights) if no model present
#   3. Build project (tests/build/)
#   4. Run GTest structural tests (test_keypoint)
#
# COCO KeypointPreview schema used by the fixture: {0, 17}
#   class 0 = background (0 keypoints)
#   class 1 = person     (17 keypoints)
# ============================================================================
set -e

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
source "$SCRIPT_DIR/test_utils.sh"

print_header "RFDETRs-CPP Keypoint Structural Tests"

# ============================================================================
# Setup
# ============================================================================
cd "$SCRIPT_DIR/keypoint"
echo "Working directory: $(pwd)"

# ============================================================================
# Install Python dependencies
# ============================================================================
print_header "Installing Dependencies"
install_uv
install_python_packages "rfdetr[onnx]"

# ============================================================================
# Prepare model
# ============================================================================
print_header "Preparing Models"
mkdir -p models

if [[ -z "$(ls models/*.onnx 2>/dev/null || true)" ]]; then
    echo "No keypoint model found — exporting RFDETRKeypointPreview (random weights)..."
    export_rfdetr_model "RFDETRKeypointPreview" "models/rfdetr-keypoint-preview.onnx"
else
    echo -e "${GREEN}Keypoint model already present in models/${NC}"
fi

# ============================================================================
# Build test suite
# ============================================================================
print_header "Building Test Suite"
cd "$SCRIPT_DIR"
./build_test.sh

# ============================================================================
# Run GTest structural tests
# ============================================================================
print_header "Running Keypoint GTests"
KP_BIN="$SCRIPT_DIR/build/tests/test_keypoint"
if [[ ! -x "$KP_BIN" ]]; then
    print_error "test_keypoint not found at $KP_BIN"
    exit 1
fi
"$KP_BIN" --gtest_color=yes

print_success "Keypoint tests completed!"
