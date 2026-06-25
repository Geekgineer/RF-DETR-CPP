#!/usr/bin/env bash
# Build-only sanity check — verifies the project compiles cleanly.
# Does NOT run any tests.  Useful in CI to catch compilation errors fast.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

ORT_DIR="${ONNXRUNTIME_DIR:-}"
if [[ -z "$ORT_DIR" ]]; then
    # Try to auto-detect a sibling onnxruntime directory
    ORT_DIR=$(find "$ROOT" -maxdepth 2 -name "onnxruntime*" -type d | sort | tail -1)
fi
if [[ -z "$ORT_DIR" ]]; then
    echo "[ERROR] Set ONNXRUNTIME_DIR or place an onnxruntime* directory next to the project root."
    exit 1
fi

BUILD_DIR="$ROOT/build_test_verify"
cmake -B "$BUILD_DIR" -S "$ROOT" \
      -DONNXRUNTIME_DIR="$ORT_DIR" \
      -DCMAKE_BUILD_TYPE=Release \
      -DBUILD_TESTS=ON \
      -DBUILD_EXAMPLES=ON \
      -DBUILD_BENCHMARKS=ON \
      -DENABLE_WARNINGS=ON \
      2>&1 | tail -20

cmake --build "$BUILD_DIR" --parallel 2>&1 | tail -20

echo ""
echo "[OK] Build passed — all targets compiled successfully."
ls "$BUILD_DIR"/{rfdetr_image_inference,rfdetr_video_inference,rfdetr_camera_inference,rfdetr_batch_inference,rfdetr_seg_image_inference,rfdetr_keypoint_image_inference} 2>/dev/null || true
rm -rf "$BUILD_DIR"
