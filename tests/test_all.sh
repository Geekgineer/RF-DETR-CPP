#!/usr/bin/env bash
# ============================================================================
# RFDETRs-CPP — Full test suite runner
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BUILD_DIR="$PROJECT_ROOT/build"

echo "================================================================"
echo "  RFDETRs-CPP Test Suite"
echo "================================================================"

# -----------------------------------------------------------------------
# Check build
# -----------------------------------------------------------------------
if [[ ! -d "$BUILD_DIR" ]]; then
    echo "[ERROR] Build directory not found: $BUILD_DIR"
    echo "  Run: cd $PROJECT_ROOT && ./build.sh -DBUILD_TESTS=ON"
    exit 1
fi

# -----------------------------------------------------------------------
# Run detection tests
# -----------------------------------------------------------------------
if [[ -x "$BUILD_DIR/compare_results" ]]; then
    echo ""
    echo "--- Detection Parity Test ---"
    bash "$SCRIPT_DIR/detection/test_detection.sh"
else
    echo "[SKIP] compare_results not built. Rebuild with -DBUILD_TESTS=ON"
fi

echo ""
echo "================================================================"
echo "  Test suite complete."
echo "================================================================"
