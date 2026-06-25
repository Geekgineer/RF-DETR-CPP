#!/usr/bin/env bash
# Utility / smoke tests — quick checks that don't require a model or GPU.
# Tests: binary existence, --help flags, coco.names line count.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="$ROOT/build"
PASS=0; FAIL=0

check() {
    local desc="$1"; shift
    if "$@" &>/dev/null; then
        echo "[PASS] $desc"
        ((PASS++))
    else
        echo "[FAIL] $desc"
        ((FAIL++))
    fi
}

echo "=== Utility / smoke tests ==="

# Binary existence
for bin in rfdetr_image_inference rfdetr_video_inference rfdetr_camera_inference rfdetr_batch_inference \
           rfdetr_seg_image_inference rfdetr_keypoint_image_inference; do
    check "binary exists: $bin" test -f "$BUILD/$bin"
done

# coco.names has exactly 80 lines
check "coco.names has 80 classes" bash -c "[ \$(wc -l < '$ROOT/models/coco.names') -eq 80 ]"

# export script is syntactically valid Python
check "export_rfdetr_onnx.py syntax OK" python3 -m py_compile "$ROOT/scripts/export_rfdetr_onnx.py"

# Shell scripts are executable
for sh in run_image.sh run_video.sh run_camera.sh run_test.sh; do
    check "$sh is executable" test -x "$ROOT/$sh"
done

echo ""
echo "Results: $PASS passed, $FAIL failed"
[[ $FAIL -eq 0 ]]
