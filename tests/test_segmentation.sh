#!/bin/bash
# ============================================================================
# RFDETRs-CPP Segmentation Structural Test Runner
# ============================================================================
# Steps:
#   1. Install rfdetr[onnx] (via uv)
#   2. Export RFDETRSegNano (random weights) if no model present
#   3. Build project (tests/build/)
#   4. Run GTest structural tests (test_seg)
# ============================================================================
set -e

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
source "$SCRIPT_DIR/test_utils.sh"

print_header "RFDETRs-CPP Segmentation Structural Tests"

# ============================================================================
# Setup
# ============================================================================
cd "$SCRIPT_DIR/segmentation"
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
    echo "No seg model found — exporting RFDETRSegNano (random weights)..."
    export_rfdetr_model "RFDETRSegNano" "models/rfdetr-seg-nano.onnx"
else
    echo -e "${GREEN}Seg model already present in models/${NC}"
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
print_header "Running Segmentation GTests"
if [[ ! -x "$SCRIPT_DIR/build/test_seg" ]]; then
    print_error "test_seg not found in tests/build/"
    exit 1
fi
"$SCRIPT_DIR/build/test_seg" --gtest_color=yes

print_success "Segmentation tests completed!"
