#!/usr/bin/env bash
# ============================================================================
# RFDETRs-CPP — Automated Benchmark Runner
# ============================================================================
# Builds and runs the benchmark suite against all .onnx models found in
# ../models/.  Results are saved to results/<timestamp>/
#
# Usage:
#   ./auto_bench.sh [--ort-dir /path/to/onnxruntime]
#                   [--input /path/to/image_or_video_or_dir]
#                   [--backend cpu|cuda|trt]
#                   [--mode image|video|batch]
#                   [--warmup 10] [--iters 100]
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# -----------------------------------------------------------------------
# Defaults
# -----------------------------------------------------------------------
ORT_DIR=""
INPUT_PATH="$PROJECT_ROOT/data/dog.jpg"
BACKEND="cpu"
MODE="image"
WARMUP=10
ITERS=100
LABELS="$PROJECT_ROOT/models/coco.names"

# -----------------------------------------------------------------------
# Parse args
# -----------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case $1 in
        --ort-dir)   ORT_DIR="$2";    shift 2 ;;
        --input)     INPUT_PATH="$2"; shift 2 ;;
        --backend)   BACKEND="$2";    shift 2 ;;
        --mode)      MODE="$2";       shift 2 ;;
        --warmup)    WARMUP="$2";     shift 2 ;;
        --iters)     ITERS="$2";      shift 2 ;;
        --labels)    LABELS="$2";     shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# -----------------------------------------------------------------------
# Locate ONNX Runtime if not provided
# -----------------------------------------------------------------------
if [[ -z "$ORT_DIR" ]]; then
    for candidate in \
        "$PROJECT_ROOT/onnxruntime-linux-x64-1.20.1" \
        "$PROJECT_ROOT/onnxruntime-linux-x64-gpu-1.20.1" \
        "$(dirname "$PROJECT_ROOT")/YOLOs-CPP/onnxruntime-linux-x64-1.20.1" \
        "/opt/onnxruntime"; do
        if [[ -f "$candidate/include/onnxruntime_cxx_api.h" ]]; then
            ORT_DIR="$candidate"
            break
        fi
    done
fi

if [[ -z "$ORT_DIR" ]]; then
    echo "[ERROR] ONNX Runtime not found. Use --ort-dir or set ONNXRUNTIME_DIR."
    exit 1
fi
echo "[INFO] ONNX Runtime: $ORT_DIR"

# -----------------------------------------------------------------------
# Build
# -----------------------------------------------------------------------
BUILD_DIR="$SCRIPT_DIR/build"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DONNXRUNTIME_DIR="$ORT_DIR" \
    -DCMAKE_SUPPRESS_REGENERATION=ON \
    > /dev/null
cmake --build . -j"$(nproc)"
echo "[INFO] Build complete."

# -----------------------------------------------------------------------
# Run against all .onnx models
# -----------------------------------------------------------------------
MODEL_DIR="$PROJECT_ROOT/models"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RESULTS_DIR="$SCRIPT_DIR/results/${TIMESTAMP}_${BACKEND}_${MODE}"
mkdir -p "$RESULTS_DIR"

BENCHMARK_BIN="$BUILD_DIR/rfdetr_benchmark"
if [[ ! -f "$BENCHMARK_BIN" ]]; then
    echo "[ERROR] Benchmark binary not found: $BENCHMARK_BIN"
    exit 1
fi

MODELS=("$MODEL_DIR"/*.onnx)
if [[ ${#MODELS[@]} -eq 0 || ! -f "${MODELS[0]}" ]]; then
    echo "[WARNING] No .onnx models found in $MODEL_DIR"
    echo "  Export a model first: python scripts/export_rfdetr_onnx.py"
    exit 0
fi

for MODEL in "${MODELS[@]}"; do
    MODEL_NAME="$(basename "$MODEL" .onnx)"
    RESULT_FILE="$RESULTS_DIR/${MODEL_NAME}.txt"
    echo ""
    echo "================================================================"
    echo "  Benchmarking: $MODEL_NAME"
    echo "================================================================"
    "$BENCHMARK_BIN" \
        "$MODEL" \
        "$INPUT_PATH" \
        "$LABELS" \
        --backend "$BACKEND" \
        --mode    "$MODE" \
        --warmup  "$WARMUP" \
        --iters   "$ITERS" \
        2>&1 | tee "$RESULT_FILE"
done

echo ""
echo "[INFO] All results saved to: $RESULTS_DIR"
