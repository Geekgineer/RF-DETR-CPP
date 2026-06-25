#!/usr/bin/env bash
# ============================================================================
# RFDETRs-CPP — Detection parity test runner
# ============================================================================
# Runs the Python ORT reference inference, then the C++ inference runner,
# then executes the GTest parity comparison.
#
# Usage: ./test_detection.sh [--build-dir ../../build]
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
BUILD_DIR="$PROJECT_ROOT/build"

while [[ $# -gt 0 ]]; do
    case $1 in
        --build-dir) BUILD_DIR="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

echo "================================================================"
echo "  RF-DETR Detection Parity Test"
echo "================================================================"

# -----------------------------------------------------------------------
# Check prerequisites
# -----------------------------------------------------------------------
MODEL_DIR="$SCRIPT_DIR/models"
IMAGES_DIR="$SCRIPT_DIR/data/images"
RESULTS_DIR="$SCRIPT_DIR/results"

if [[ -z "$(ls "$MODEL_DIR"/*.onnx 2>/dev/null)" ]]; then
    echo "[ERROR] No .onnx models in $MODEL_DIR"
    echo "  Export one first: python $PROJECT_ROOT/scripts/export_rfdetr_onnx.py"
    exit 1
fi

if [[ -z "$(ls "$IMAGES_DIR"/*.{jpg,jpeg,png,bmp} 2>/dev/null || true)" ]]; then
    echo "[ERROR] No test images in $IMAGES_DIR"
    echo "  Place at least one image there and retry."
    exit 1
fi

mkdir -p "$RESULTS_DIR"

# -----------------------------------------------------------------------
# Step 1: Python ORT reference
# -----------------------------------------------------------------------
echo ""
echo "[1/3] Python ORT reference inference..."
cd "$SCRIPT_DIR"
python3 inference_rfdetr_python.py
echo "      Done: $RESULTS_DIR/results_python.json"

# -----------------------------------------------------------------------
# Step 2: C++ inference runner
# -----------------------------------------------------------------------
echo ""
echo "[2/3] C++ inference runner..."
if [[ ! -x "$BUILD_DIR/inference_rfdetr_cpp" ]]; then
    echo "[ERROR] inference_rfdetr_cpp not found in $BUILD_DIR"
    echo "  Rebuild with: cmake .. -DBUILD_TESTS=ON && make -j\$(nproc)"
    exit 1
fi
"$BUILD_DIR/inference_rfdetr_cpp" "$MODEL_DIR" "$IMAGES_DIR"
echo "      Done: $RESULTS_DIR/results_cpp.json"

# -----------------------------------------------------------------------
# Step 3: GTest parity comparison
# -----------------------------------------------------------------------
echo ""
echo "[3/3] GTest parity comparison (tol: bbox ±2px, conf ±0.002)..."
if [[ ! -x "$BUILD_DIR/compare_results" ]]; then
    echo "[ERROR] compare_results not found in $BUILD_DIR"
    exit 1
fi
"$BUILD_DIR/compare_results" --gtest_color=yes
echo ""
echo "================================================================"
echo "  Detection parity test PASSED."
echo "================================================================"
