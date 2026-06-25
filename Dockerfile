# ============================================================================
# RFDETRs-CPP Docker Image (GPU + CPU)
# ============================================================================
# Multi-stage build: builder produces binaries, runtime is slim.
#
# Build: docker build -t rfdetrs-cpp .
# Run:   docker run --rm -it rfdetrs-cpp
# GPU:   docker run --gpus all --rm -it rfdetrs-cpp
# ============================================================================

# ---- Stage 1: Builder -------------------------------------------------------
FROM nvidia/cuda:12.2.0-devel-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake git wget ca-certificates \
    libopencv-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

ARG ONNXRUNTIME_VERSION=1.20.1
RUN wget -q \
    "https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-linux-x64-gpu-${ONNXRUNTIME_VERSION}.tgz" \
    && tar -xzf "onnxruntime-linux-x64-gpu-${ONNXRUNTIME_VERSION}.tgz" \
    && mv "onnxruntime-linux-x64-gpu-${ONNXRUNTIME_VERSION}" /opt/onnxruntime \
    && rm "onnxruntime-linux-x64-gpu-${ONNXRUNTIME_VERSION}.tgz"

COPY . /build/src
WORKDIR /build/src

RUN mkdir -p build && cd build \
    && cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DONNXRUNTIME_DIR=/opt/onnxruntime \
    && make -j"$(nproc)"

# ---- Stage 2: Runtime -------------------------------------------------------
FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    libopencv-core4.5d libopencv-imgproc4.5d \
    libopencv-imgcodecs4.5d libopencv-videoio4.5d libopencv-highgui4.5d \
    libgtk-3-0 libcanberra-gtk3-module \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/onnxruntime/lib /opt/onnxruntime/lib
COPY --from=builder /build/src/build/rfdetr_image_inference  /app/
COPY --from=builder /build/src/build/rfdetr_video_inference  /app/
COPY --from=builder /build/src/build/rfdetr_camera_inference /app/
COPY --from=builder /build/src/build/rfdetr_batch_inference  /app/
COPY --from=builder /build/src/models /app/models
COPY --from=builder /build/src/data   /app/data

ENV LD_LIBRARY_PATH=/opt/onnxruntime/lib:$LD_LIBRARY_PATH
WORKDIR /app

ENV INFERENCE_TARGET=rfdetr_image_inference
ENV MODEL_PATH=models/rfdetr-base.onnx
ENV INPUT_PATH=data/dog.jpg
ENV LABELS_PATH=models/coco.names

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD test -f /app/${INFERENCE_TARGET} || exit 1

CMD ./${INFERENCE_TARGET} ${MODEL_PATH} ${INPUT_PATH} ${LABELS_PATH}

LABEL maintainer="RFDETRs-CPP <https://github.com/Geekgineer/rf-detr-cpp>"
LABEL version="1.0.0"
LABEL description="C++ ONNX Runtime inference for RF-DETR detection models"
