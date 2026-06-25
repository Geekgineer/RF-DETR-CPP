#!/usr/bin/env bash
# ============================================================================
# RFDETRs-CPP — Detection parity test (CI-aware)
# ============================================================================
# Exports a random-weight RF-DETR Nano model if no model is present in
# tests/detection/models/, then runs the full 3-step parity test:
#   1. Python ORT reference inference  → results_python.json
#   2. C++ inference runner            → results_cpp.json
#   3. GTest parity comparison         → compare_results
#
# Usage:
#   bash tests/test_detection.sh [--build-dir /path/to/build]
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

MODEL_DIR="$SCRIPT_DIR/detection/models"
mkdir -p "$MODEL_DIR"

# -----------------------------------------------------------------------
# Export model if none present (CI: random weights, fast)
# -----------------------------------------------------------------------
if [[ -z "$(ls "$MODEL_DIR"/*.onnx 2>/dev/null || true)" ]]; then
    echo "[CI] No model found — exporting RFDETRNano (random weights)..."
    python3 - "$MODEL_DIR" <<'PYEOF'
import sys, shutil, tempfile, glob
from rfdetr import RFDETRNano
out_dir = sys.argv[1]
m = RFDETRNano(pretrain_weights=None)
with tempfile.TemporaryDirectory() as tmp:
    m.export(output_dir=tmp)
    srcs = glob.glob(tmp + '/**/*.onnx', recursive=True)
    if not srcs:
        raise FileNotFoundError('No .onnx produced by export')
    shutil.copy2(srcs[0], out_dir + '/rfdetr-nano.onnx')
    print(f'Exported: {out_dir}/rfdetr-nano.onnx')
PYEOF
fi

# -----------------------------------------------------------------------
# Run parity test
# -----------------------------------------------------------------------
bash "$SCRIPT_DIR/detection/test_detection.sh" --build-dir "$BUILD_DIR"
