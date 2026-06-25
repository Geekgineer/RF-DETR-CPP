# RFDETRs-CPP

**Modern C++ inference library for RF-DETR v1.8+ — detection, segmentation, and keypoint tasks.**

RFDETRs-CPP is a header-only C++ ONNX Runtime inference library for RF-DETR models. It mirrors the
structure of [YOLOs-CPP](https://github.com/Geekgineer/YOLOs-CPP) and supports all three RF-DETR task types
with CPU, CUDA, and TensorRT backends.

## Features

- **Detection** — `RFDETRDetector` with logit-space pre-filter and `nth_element` top-K
- **Segmentation** — `RFDETRSegDetector` with bilinear upsample + binary mask output
- **Keypoint** — `RFDETRKeypointDetector` with class-major slot decoding (GroupPose NLL-Cholesky)
- **Backends** — CPU · CUDA · TensorRT via `BackendType` enum
- **Header-only tasks** — include `rfdetrs/rfdetrs.hpp` and you're done

## Quick Start

```bash
# Clone and build
git clone https://github.com/Geekgineer/RF-DETR-CPP.git
cd RF-DETR-CPP
./build.sh

# Run detection on an image
./build/rfdetr_image_inference rfdetr-nano.onnx dog.jpg models/coco.names 0.5 300 cpu
```

## ONNX Contracts

| Task        | Outputs |
|-------------|---------|
| Detection   | `dets [1,N,4]` · `labels [1,N,C]` |
| Segmentation | `dets [1,N,4]` · `labels [1,N,C]` · `masks [1,N,H/4,W/4]` |
| Keypoint    | `dets [1,N,4]` · `labels [1,N,C]` · `keypoints [1,N,C*max_K,8]` |

## License

AGPL-3.0-only — see [LICENSE](https://github.com/Geekgineer/RF-DETR-CPP/blob/main/LICENSE).
