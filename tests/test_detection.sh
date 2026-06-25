#!/bin/bash
# ============================================================================
# RFDETRs-CPP Detection Parity Test Runner
# ============================================================================
# Steps:
#   1. Install rfdetr[onnx] + onnxruntime (via uv)
#   2. Export RFDETRNano (random weights) if no model present
#   3. Build project (tests/build/)
#   4. Run Python ORT reference  → detection/results/results_python.json
#   5. Run C++ inference runner  → detection/results/results_cpp.json
#   6. Run GTest parity comparison
# ============================================================================
set -e

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
source "$SCRIPT_DIR/test_utils.sh"

print_header "RFDETRs-CPP Detection Parity Test"

# ============================================================================
# Setup
# ============================================================================
cd "$SCRIPT_DIR/detection"
echo "Working directory: $(pwd)"

# ============================================================================
# Install Python dependencies
# ============================================================================
print_header "Installing Dependencies"
install_uv
install_python_packages "rfdetr[onnx]" onnxruntime

# ============================================================================
# Prepare model
# ============================================================================
print_header "Preparing Models"
mkdir -p models

if [[ -z "$(ls models/*.onnx 2>/dev/null || true)" ]]; then
    echo "No model found — exporting RFDETRNano (random weights)..."
    cd "$SCRIPT_DIR/detection"
    export_rfdetr_model "RFDETRNano" "models/rfdetr-nano.onnx"
else
    echo -e "${GREEN}Model already present in models/${NC}"
fi

# ============================================================================
# Build test suite
# ============================================================================
print_header "Building Test Suite"
cd "$SCRIPT_DIR"
./build_test.sh

# ============================================================================
# Generate Python reference
# ============================================================================
print_header "Generating Python Reference"
cd "$SCRIPT_DIR/detection"
echo "Running Python ORT inference..."
python3 inference_rfdetr_python.py || {
    print_error "Python reference inference failed"
    exit 1
}
print_success "Python reference generated: results/results_python.json"

# ============================================================================
# Run C++ inference runner
# ============================================================================
print_header "Running C++ Inference"
if [[ ! -x "$SCRIPT_DIR/build/inference_rfdetr_cpp" ]]; then
    print_error "inference_rfdetr_cpp not found in tests/build/"
    exit 1
fi
"$SCRIPT_DIR/build/inference_rfdetr_cpp" models/ data/images/
print_success "C++ inference done: results/results_cpp.json"

# ============================================================================
# Run GTest parity comparison
# ============================================================================
print_header "Running Parity Tests (tol: bbox ±2px, conf ±0.002)"
"$SCRIPT_DIR/build/compare_results" --gtest_color=yes

print_success "Detection parity tests completed!"
