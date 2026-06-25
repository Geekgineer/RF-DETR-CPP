#!/usr/bin/env bash
# ============================================================================
# RFDETRs-CPP — Keypoint structural tests (CI-aware)
# ============================================================================
# Exports a random-weight RFDETRKeypointPreview model if none is present in
# tests/keypoint/models/, then runs the GTest structural test suite.
#
# COCO schema passed to the keypoint fixture: {0, 17}
#   class 0 = background (0 keypoints)
#   class 1 = person     (17 keypoints)
#
# Usage:
#   bash tests/test_keypoint.sh [--build-dir /path/to/build]
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="${BUILD_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)/build}"

while [[ $# -gt 0 ]]; do
    case $1 in
        --build-dir) BUILD_DIR="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

MODEL_DIR="$SCRIPT_DIR/keypoint/models"
mkdir -p "$MODEL_DIR"

# -----------------------------------------------------------------------
# Export keypoint model if absent (CI: random weights)
# -----------------------------------------------------------------------
if [[ -z "$(ls "$MODEL_DIR"/*.onnx 2>/dev/null || true)" ]]; then
    echo "[CI] No keypoint model found — exporting RFDETRKeypointPreview (random weights)..."
    python3 - "$MODEL_DIR" <<'PYEOF'
import sys, shutil, tempfile, glob
from rfdetr import RFDETRKeypointPreview
out_dir = sys.argv[1]
m = RFDETRKeypointPreview(pretrain_weights=None)
with tempfile.TemporaryDirectory() as tmp:
    m.export(output_dir=tmp)
    srcs = glob.glob(tmp + '/**/*.onnx', recursive=True)
    if not srcs:
        raise FileNotFoundError('No .onnx produced by RFDETRKeypointPreview export')
    shutil.copy2(srcs[0], out_dir + '/rfdetr-keypoint-preview.onnx')
    print(f'Exported: {out_dir}/rfdetr-keypoint-preview.onnx')
PYEOF
fi

# -----------------------------------------------------------------------
# Run GTest structural tests
# -----------------------------------------------------------------------
echo ""
echo "================================================================"
echo "  RF-DETR Keypoint Structural Tests"
echo "================================================================"

if [[ ! -x "$BUILD_DIR/test_keypoint" ]]; then
    echo "[ERROR] test_keypoint not found in $BUILD_DIR"
    echo "  Rebuild with: cmake .. -DBUILD_TESTS=ON && make -j\$(nproc)"
    exit 1
fi

"$BUILD_DIR/test_keypoint" --gtest_color=yes

echo ""
echo "================================================================"
echo "  Keypoint tests PASSED."
echo "================================================================"
