#!/usr/bin/env bash
# ============================================================================
# RFDETRs-CPP — Segmentation structural tests (CI-aware)
# ============================================================================
# Exports a random-weight RFDETRSegNano model if none is present in
# tests/segmentation/models/, then runs the GTest structural test suite.
#
# Usage:
#   bash tests/test_segmentation.sh [--build-dir /path/to/build]
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

MODEL_DIR="$SCRIPT_DIR/segmentation/models"
mkdir -p "$MODEL_DIR"

# -----------------------------------------------------------------------
# Export segmentation model if absent (CI: random weights)
# -----------------------------------------------------------------------
if [[ -z "$(ls "$MODEL_DIR"/*.onnx 2>/dev/null || true)" ]]; then
    echo "[CI] No seg model found — exporting RFDETRSegNano (random weights)..."
    python3 - "$MODEL_DIR" <<'PYEOF'
import sys, shutil, tempfile, glob
from rfdetr import RFDETRSegNano
out_dir = sys.argv[1]
m = RFDETRSegNano(pretrain_weights=None)
with tempfile.TemporaryDirectory() as tmp:
    m.export(output_dir=tmp)
    srcs = glob.glob(tmp + '/**/*.onnx', recursive=True)
    if not srcs:
        raise FileNotFoundError('No .onnx produced by RFDETRSegNano export')
    shutil.copy2(srcs[0], out_dir + '/rfdetr-seg-nano.onnx')
    print(f'Exported: {out_dir}/rfdetr-seg-nano.onnx')
PYEOF
fi

# -----------------------------------------------------------------------
# Run GTest structural tests
# -----------------------------------------------------------------------
echo ""
echo "================================================================"
echo "  RF-DETR Segmentation Structural Tests"
echo "================================================================"

if [[ ! -x "$BUILD_DIR/test_seg" ]]; then
    echo "[ERROR] test_seg not found in $BUILD_DIR"
    echo "  Rebuild with: cmake .. -DBUILD_TESTS=ON && make -j\$(nproc)"
    exit 1
fi

"$BUILD_DIR/test_seg" --gtest_color=yes

echo ""
echo "================================================================"
echo "  Segmentation tests PASSED."
echo "================================================================"
