#!/bin/bash
# ============================================================================
# RFDETRs-CPP build script
# Mirrors YOLOs-CPP/build.sh — auto-downloads ONNX Runtime if absent.
#
# Usage:
#   ./build.sh [ORT_VERSION] [GPU=0|1]
#
# Examples:
#   ./build.sh              # CPU build with ORT 1.20.1
#   ./build.sh 1.20.1 1     # GPU build with ORT 1.20.1
# ============================================================================
set -euo pipefail

CURRENT_DIR=$(cd "$(dirname "$0")" && pwd)

ORT_VERSION="${1:-1.20.1}"
ORT_GPU="${2:-0}"

platform=$(uname -s)
arch=$(uname -m)

case "$platform" in
    Darwin*) ORT_PLATFORM="osx";   EXT="tgz" ;;
    Linux*)  ORT_PLATFORM="linux"; EXT="tgz" ;;
    MINGW*)  ORT_PLATFORM="win";   EXT="zip" ;;
    *) echo "Unsupported platform: $platform"; exit 1 ;;
esac

case "$arch" in
    aarch64|arm64) ORT_ARCH="aarch64" ;;
    x86_64)        ORT_ARCH="x64"     ;;
    arm*)          ORT_ARCH="arm"     ;;
    i*86)          ORT_ARCH="x86"     ;;
    *) echo "Unsupported arch: $arch"; exit 1 ;;
esac

ORT_FILE="onnxruntime-${ORT_PLATFORM}-${ORT_ARCH}"
ORT_DIR="${CURRENT_DIR}/onnxruntime-${ORT_PLATFORM}-${ORT_ARCH}"

if [[ "$ORT_GPU" -eq 1 ]]; then
    ORT_FILE="${ORT_FILE}-gpu"
    ORT_DIR="${ORT_DIR}-gpu"
fi

ORT_FILE="${ORT_FILE}-${ORT_VERSION}.${EXT}"
ORT_DIR="${ORT_DIR}-${ORT_VERSION}"
ORT_URL="https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/${ORT_FILE}"

if [ ! -d "$ORT_DIR" ]; then
    echo "Downloading ONNX Runtime ${ORT_VERSION} from ${ORT_URL} ..."
    curl -L -C - -o "${ORT_FILE}" "${ORT_URL}"
    echo "Extracting ..."
    if [[ "$EXT" == "tgz" ]]; then
        tar -zxf "${ORT_FILE}" -C "${CURRENT_DIR}"
    else
        unzip "${ORT_FILE}" -d "${CURRENT_DIR}"
    fi
    rm -f "${ORT_FILE}"
else
    echo "ONNX Runtime already present at ${ORT_DIR}."
fi

BUILD_DIR="${CURRENT_DIR}/build"
mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

echo "Configuring CMake ..."
cmake .. \
    -DONNXRUNTIME_DIR="${ORT_DIR}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_FLAGS_RELEASE="-O3 -march=native"

echo "Building ..."
cmake --build . -- -j"$(nproc)"

echo ""
echo "Build complete. Binaries are in: ${BUILD_DIR}"
echo ""
echo "Quick start:"
echo "  ${BUILD_DIR}/rfdetr_image_inference <model.onnx> <image.jpg> [labels.names]"
