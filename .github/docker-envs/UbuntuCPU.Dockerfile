# ============================================================================
# RFDETRs-CPP — Ubuntu 22.04 CPU build environment
# ============================================================================
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

# System build tools + OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    libopencv-dev \
    python3 \
    python3-pip \
    wget \
    ca-certificates \
    git \
    && rm -rf /var/lib/apt/lists/*

# ONNX Runtime (CPU) — same version as the project build
ARG ORT_VERSION=1.20.1
RUN wget -q "https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/onnxruntime-linux-x64-${ORT_VERSION}.tgz" \
         -O /tmp/ort.tgz \
    && mkdir -p /opt/onnxruntime \
    && tar -xf /tmp/ort.tgz -C /opt/onnxruntime --strip-components=1 \
    && rm /tmp/ort.tgz

ENV ONNXRUNTIME_DIR=/opt/onnxruntime
ENV LD_LIBRARY_PATH=/opt/onnxruntime/lib:${LD_LIBRARY_PATH:-}

# Python inference deps (for detection parity test and model export)
RUN pip3 install --quiet "rfdetr[onnx]" onnxruntime

WORKDIR /workspace
