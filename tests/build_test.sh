#!/bin/bash
# ============================================================================
# RFDETRs-CPP Test Build Script
# ============================================================================
# Downloads ONNX Runtime if absent and builds the full project (including
# tests) into tests/build/. Called by individual task test scripts.
#
# Usage: ./build_test.sh [ONNXRUNTIME_VERSION] [ONNXRUNTIME_GPU]
#
# In CI the workflow extracts ORT into tests/ before calling the test scripts,
# so the download step is skipped automatically.
# ============================================================================
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

ONNXRUNTIME_VERSION="${1:-1.20.1}"
ONNXRUNTIME_GPU="${2:-0}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# ============================================================================
# Platform detection
# ============================================================================
detect_platform() {
    local platform
    platform=$(uname -s)
    local arch
    arch=$(uname -m)

    case "$platform" in
        Darwin*) ONNXRUNTIME_PLATFORM="osx";   ARCHIVE_EXT="tgz" ;;
        Linux*)  ONNXRUNTIME_PLATFORM="linux";  ARCHIVE_EXT="tgz" ;;
        MINGW*)  ONNXRUNTIME_PLATFORM="win";    ARCHIVE_EXT="zip" ;;
        *) echo -e "${RED}Unsupported platform: $platform${NC}"; exit 1 ;;
    esac

    case "$arch" in
        aarch64|arm64) ONNXRUNTIME_ARCH="aarch64" ;;
        x86_64)        ONNXRUNTIME_ARCH="x64"     ;;
        arm*)          ONNXRUNTIME_ARCH="arm"      ;;
        *) echo -e "${RED}Unsupported arch: $arch${NC}"; exit 1 ;;
    esac

    ONNXRUNTIME_BASENAME="onnxruntime-${ONNXRUNTIME_PLATFORM}-${ONNXRUNTIME_ARCH}"
    if [[ "$ONNXRUNTIME_GPU" -eq 1 ]]; then
        ONNXRUNTIME_BASENAME="${ONNXRUNTIME_BASENAME}-gpu"
    fi

    ONNXRUNTIME_FILE="${ONNXRUNTIME_BASENAME}-${ONNXRUNTIME_VERSION}.${ARCHIVE_EXT}"
    ONNXRUNTIME_DIR="${SCRIPT_DIR}/${ONNXRUNTIME_BASENAME}-${ONNXRUNTIME_VERSION}"
    ONNXRUNTIME_URL="https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/${ONNXRUNTIME_FILE}"
}

# ============================================================================
# Download ONNX Runtime (skipped if already present — CI extracts it first)
# ============================================================================
download_onnxruntime() {
    if [[ -d "$ONNXRUNTIME_DIR" ]]; then
        echo -e "${GREEN}ONNX Runtime found: $ONNXRUNTIME_DIR${NC}"
        return
    fi

    echo -e "${YELLOW}Downloading ONNX Runtime from ${ONNXRUNTIME_URL}...${NC}"
    curl -L -C - -o "${SCRIPT_DIR}/${ONNXRUNTIME_FILE}" "$ONNXRUNTIME_URL" || {
        echo -e "${RED}Download failed${NC}"; exit 1
    }

    echo -e "${YELLOW}Extracting...${NC}"
    if [[ "$ARCHIVE_EXT" == "tgz" ]]; then
        tar -xzf "${SCRIPT_DIR}/${ONNXRUNTIME_FILE}" -C "$SCRIPT_DIR"
    else
        unzip "${SCRIPT_DIR}/${ONNXRUNTIME_FILE}" -d "$SCRIPT_DIR"
    fi
    rm -f "${SCRIPT_DIR}/${ONNXRUNTIME_FILE}"
    echo -e "${GREEN}Extracted to: $ONNXRUNTIME_DIR${NC}"
}

# ============================================================================
# Build project (all targets + tests)
# ============================================================================
build_project() {
    local build_dir="${SCRIPT_DIR}/build"
    mkdir -p "$build_dir"
    cd "$build_dir"

    echo -e "${YELLOW}Configuring CMake...${NC}"
    cmake "${SCRIPT_DIR}/.." \
        -DONNXRUNTIME_DIR="${ONNXRUNTIME_DIR}" \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_TESTS=ON \
        -DCMAKE_CXX_FLAGS_RELEASE="-O3"

    echo -e "${YELLOW}Building...${NC}"
    cmake --build . -- -j"$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)"

    echo -e "${GREEN}Build complete!${NC}"
    echo "  CLIs  : $build_dir/rfdetr_*"
    echo "  GTests: $build_dir/tests/"
    cd "$SCRIPT_DIR"
}

# ============================================================================
# Main
# ============================================================================
echo "============================================"
echo "  RFDETRs-CPP Test Build"
echo "============================================"
echo "  ONNX Runtime: $ONNXRUNTIME_VERSION"
echo "  GPU:          $ONNXRUNTIME_GPU"
echo "============================================"

detect_platform
download_onnxruntime
build_project

echo ""
echo -e "${GREEN}Build complete!${NC}"
