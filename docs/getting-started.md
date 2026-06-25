# Getting Started

## Prerequisites

- CMake ≥ 3.16
- OpenCV ≥ 4.x (`libopencv-dev`)
- ONNX Runtime ≥ 1.18 ([download](https://github.com/microsoft/onnxruntime/releases))
- C++17 compiler (GCC 11+, Clang 14+)

## Build

```bash
git clone https://github.com/Geekgineer/RF-DETR-CPP.git
cd RF-DETR-CPP
./build.sh
```

Or manually:

```bash
cmake -B build \
  -DONNXRUNTIME_DIR=/path/to/onnxruntime-linux-x64-1.20.1 \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel $(nproc)
```

## Export a Model

```bash
pip install rfdetr[onnx]
python scripts/export_rfdetr_onnx.py --variant nano --out-dir models/
```

## Run Inference

```bash
# Detection
./build/rfdetr_image_inference models/rfdetr-nano.onnx dog.jpg models/coco.names

# Segmentation
./build/rfdetr_seg_image_inference models/rfdetr-seg-nano.onnx dog.jpg models/coco.names

# Keypoint (COCO: class 0=bg, class 1=person with 17 kps)
./build/rfdetr_keypoint_image_inference models/rfdetr-keypoint-preview.onnx dog.jpg "" 0.5 300 cpu 0,17
```
