# RFDETRs-CPP

C++ ONNX Runtime inference library for RF-DETR — CPU, CUDA, and TensorRT backends behind a single API.

```
./rfdetr_image_inference models/rfdetr-base.onnx data/dog.jpg models/coco.names
```

---

## What it is

RFDETRs-CPP wraps RF-DETR's exported ONNX graph in a zero-dependency C++ detector class that mirrors the YOLOs-CPP API. It handles the full pipeline: stretch-resize + ImageNet normalisation, ORT session management with provider fallback, and the flat top-K postprocess that matches `rfdetr.models.postprocess._select_topk` exactly.

**Postprocess is NOT YOLO:** sigmoid over all logits → top-K over flattened `[N×C]` — no per-query argmax, no NMS, no objectness gate.

---

## Requirements

| Dependency | Version | Notes |
|---|---|---|
| CMake | ≥ 3.16 | |
| C++ compiler | C++17 | GCC 9+, Clang 9+, MSVC 2019+ |
| OpenCV | ≥ 4.5 | core, imgproc, imgcodecs, videoio, highgui |
| ONNX Runtime | 1.20.1 | CPU EP always; CUDA / TRT optional |
| Python + rfdetr | 1.8.x | Export only; not required at inference time |
| CUDA Toolkit | ≥ 12.0 | Optional, for GPU EPs |

Parity tests add GoogleTest 1.14 and nlohmann/json 3.11.3 via CMake `FetchContent`.

---

## Quickstart

### 1. Export the ONNX model

```bash
# create a virtualenv and install rfdetr
uv venv .venv && uv pip install rfdetr onnx

# export RF-DETR Base (downloads COCO pretrain weights automatically)
.venv/bin/python scripts/export_rfdetr_onnx.py --variant base --out-dir models/

# other variants: nano  small  base  medium  large
```

### 2. Build

```bash
# CPU build
cmake -B build -DCMAKE_BUILD_TYPE=Release \
      -DONNXRUNTIME_DIR=/path/to/onnxruntime
cmake --build build -j$(nproc)

# with parity tests
cmake -B build -DCMAKE_BUILD_TYPE=Release \
      -DONNXRUNTIME_DIR=/path/to/onnxruntime \
      -DBUILD_TESTS=ON
cmake --build build -j$(nproc)
```

### 3. Run

```bash
# image → annotated JPEG + JSON sidecar
./build/rfdetr_image_inference models/rfdetr-base.onnx data/dog.jpg \
    models/coco.names 0.5 300 cpu

# video (threaded pipeline)
./build/rfdetr_video_inference models/rfdetr-base.onnx data/video.mp4 \
    data/video_out.mp4 models/coco.names

# live camera (press q to quit, s to snapshot)
./build/rfdetr_camera_inference models/rfdetr-base.onnx 0 models/coco.names

# batch directory
./build/rfdetr_batch_inference models/rfdetr-base.onnx data/images/ models/coco.names
```

> `run_image.sh`, `run_video.sh`, `run_camera.sh` in the project root are one-liner wrappers that `cd build/` and forward all arguments.

---

## C++ API

```cpp
#include "rfdetrs/rfdetrs.hpp"

using namespace rfdetrs;
using namespace rfdetrs::det;

// Construction
//   modelPath   : path to exported rfdetr-*.onnx
//   labelsPath  : path to coco.names (empty string → use class IDs)
//   backend     : CPU (default), CUDA, TensorRT
RFDETRDetector det("models/rfdetr-base.onnx",
                   "models/coco.names",
                   BackendType::CPU);

// Inference
//   confThreshold : minimum sigmoid score to keep (default 0.5)
//   topK          : max (query, class) pairs to evaluate (default 300)
cv::Mat img = cv::imread("image.jpg");
std::vector<Detection> results = det.detect(img, 0.5f, 300);

// Inspect
for (const auto& d : results) {
    // d.box     → BoundingBox { x, y, width, height } in pixel coords
    // d.conf    → sigmoid score in [0, 1]
    // d.classId → COCO class index
}

// Visualise
det.drawDetections(img, results);
cv::imwrite("out.jpg", img);
```

BackendType falls back gracefully — `CUDA` falls back to CPU if the CUDA EP is unavailable; `TensorRT` falls back CUDA→CPU.

---

## Supported model variants

All five detection variants are Apache-2.0 licensed. XL and 2XL (PML 1.0) are excluded.

| Variant | Resolution | Patch size | Queries | Export flag |
|---|---|---|---|---|
| Nano   | 384×384 | 16 | 300 | `--variant nano` |
| Small  | 512×512 | 16 | 300 | `--variant small` |
| Base   | 560×560 | 14 | 300 | `--variant base` |
| Medium | 576×576 | 16 | 300 | `--variant medium` |
| Large  | 704×704 | 16 | 300 | `--variant large` |

**ONNX contract:** input `float32[1, 3, H, W]` (static shape). Outputs `dets [1, 300, 4]` cxcywh normalised and `labels [1, 300, 91]` raw logits. Outputs identified by shape (last-dim == 4 → boxes), not by name. Requires ONNX opset ≥ 16.

---

## Inference pipeline

```
Image (BGR) → Preprocess → ORT Session → Postprocess → Detections
               stretch         CPU/CUDA/    sigmoid         cxcywh
               resize +        TensorRT     top-K over        → xyxy
               ImageNet                     N×C flat          + scale
               normalize
```

---

## Benchmarks

Measured with `rfdetr_benchmark` — 3 warmup runs, 10 timed runs. CPU: Intel Core i7, single process.

| Model | Backend | Min | Avg | Max | FPS |
|---|---|---|---|---|---|
| rfdetr-base | CPU | 388 ms | 399 ms | 442 ms | 2.51 |

```bash
# build standalone benchmark
cmake -B benchmarks/build -S benchmarks \
      -DONNXRUNTIME_DIR=/path/to/onnxruntime -DCMAKE_BUILD_TYPE=Release
cmake --build benchmarks/build -j$(nproc)

# image benchmark — 10 warmup, 200 timed runs
./benchmarks/build/rfdetr_benchmark models/rfdetr-base.onnx data/dog.jpg \
    models/coco.names --backend cpu --mode image --warmup 10 --iters 200

# automated sweep across all .onnx in models/
./benchmarks/auto_bench.sh --backend cpu --mode image
```

---

## Parity tests

The test suite runs the same ONNX graph on both Python ORT and the C++ implementation, then compares outputs at tight tolerances.

| Assertion | Tolerance | Rationale |
|---|---|---|
| Detection count | exact | Same top-K, same threshold |
| Bounding box | ±2 px | `int()` rounding of identical float coords |
| Confidence score | ±0.002 | float32 sigmoid precision |

```bash
# build with tests
cmake -B build -DBUILD_TESTS=ON -DONNXRUNTIME_DIR=/path/to/ort
cmake --build build -j$(nproc)

# place model + test image, then run all three steps
cp models/rfdetr-base.onnx    tests/detection/models/
cp data/dog.jpg               tests/detection/data/images/

.venv/bin/python tests/detection/inference_rfdetr_python.py
./build/tests/inference_rfdetr_cpp
./build/tests/compare_results --gtest_color=yes
```

Result: **4 / 4 tests passed** — RF-DETR Base, dog.jpg (class=18 dog, conf=0.977).

---

## Docker

```bash
# GPU build (nvidia/cuda:12.2.0-devel-ubuntu22.04 base)
docker build -t rfdetrs-cpp .
docker run --gpus all --rm rfdetrs-cpp \
    ./rfdetr_image_inference models/rfdetr-base.onnx data/dog.jpg models/coco.names

# CPU only (ubuntu:22.04 base)
docker build -t rfdetrs-cpp:cpu -f Dockerfile.cpu .
docker run --rm rfdetrs-cpp:cpu
```

---

## Project layout

```
rf-detr-cpp/
├── include/rfdetrs/
│   ├── rfdetrs.hpp                     master include
│   ├── core/
│   │   ├── types.hpp                   BoundingBox, Detection
│   │   ├── preprocessing.hpp           stretch resize + ImageNet normalize → CHW blob
│   │   ├── drawing.hpp                 drawBoundingBox
│   │   └── utils.hpp                   getClassNames, generateColors, sigmoid, clamp
│   ├── tasks/
│   │   └── detection.hpp               OrtSessionBase + RFDETRDetector
│   └── tools/
│       └── BoundedThreadSafeQueue.hpp  bounded producer-consumer queue for camera
├── src/
│   ├── rfdetr_image_inference.cpp
│   ├── rfdetr_video_inference.cpp
│   ├── rfdetr_camera_inference.cpp
│   └── rfdetr_batch_inference.cpp
├── examples/
│   ├── utils.hpp
│   ├── example_image_det.cpp
│   ├── example_video_det.cpp
│   └── example_camera_det.cpp
├── benchmarks/
│   ├── rfdetr_benchmark.cpp
│   ├── auto_bench.sh
│   └── CMakeLists.txt
├── tests/detection/
│   ├── inference_rfdetr_python.py      ORT-python reference
│   ├── inference_rfdetr_cpp.cpp        C++ runner → results_cpp.json
│   └── compare_results.cpp             GTest parity (±2px, ±0.002)
├── models/  coco.names  rfdetr-*.onnx
├── scripts/ export_rfdetr_onnx.py
├── Dockerfile  Dockerfile.cpu
├── run_image.sh  run_video.sh  run_camera.sh
└── CMakeLists.txt  build.sh
```

---

## CMake options

| Option | Default | Description |
|---|---|---|
| `ONNXRUNTIME_DIR` | — | Path to ORT installation (required) |
| `BUILD_TESTS` | OFF | Build GTest parity suite |
| `BUILD_EXAMPLES` | OFF | Build example applications |
| `BUILD_BENCHMARKS` | OFF | Build benchmark suite |

---

## License

GNU Affero General Public License v3.0 (AGPL-3.0).

RF-DETR detection models (Nano · Small · Base · Medium · Large) are Apache-2.0 (upstream). The XL and 2XL variants are licensed under PML 1.0 and are not included.
