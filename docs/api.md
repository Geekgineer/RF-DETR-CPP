# API Reference

## Detection

```cpp
#include "rfdetrs/rfdetrs.hpp"
using namespace rfdetrs;

RFDETRDetector detector("rfdetr-nano.onnx", "coco.names", BackendType::CPU);
std::vector<Detection> dets = detector.detect(image, /*conf=*/0.5f, /*topK=*/300);
detector.drawDetections(image, dets);
```

## Segmentation

```cpp
using namespace rfdetrs::seg;

RFDETRSegDetector detector("rfdetr-seg-nano.onnx", "", BackendType::CPU);
std::vector<SegmentationDetection> dets = detector.detect(image, 0.5f, 300);
// dets[i].mask is CV_8UC1, same size as input image
```

## Keypoint

```cpp
using namespace rfdetrs::kp;

// COCO schema: class 0=bg (0 kps), class 1=person (17 kps)
RFDETRKeypointDetector detector("rfdetr-keypoint-preview.onnx", "", {0, 17}, BackendType::CPU);
std::vector<KeypointDetection> dets = detector.detect(image, 0.5f, 300);
// dets[i].keypoints is a vector of Keypoint{x, y, confidence, visible}
```

## Backend Selection

```cpp
BackendType::CPU        // default, always available
BackendType::CUDA       // requires CUDA ORT build
BackendType::TensorRT   // requires TensorRT ORT build
```
