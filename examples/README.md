# RFDETRs-CPP Examples

Minimal, self-contained programs demonstrating the `RFDETRDetector` API for each input modality.

## Available examples

| Example | Input | Description |
|---------|-------|-------------|
| `example_image_det` | Image / directory | Detection on static images; saves annotated output |
| `example_video_det` | Video file | Detection on every frame; writes annotated MP4 |
| `example_camera_det` | Live camera | Real-time detection with FPS overlay; press `s` to snapshot, `q` to quit |

## Building

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release \
      -DONNXRUNTIME_DIR=/path/to/onnxruntime \
      -DBUILD_EXAMPLES=ON
cmake --build build -j$(nproc)
```

Binaries land in `build/examples/`.

## Running

### Using shell scripts

```bash
./examples/run_image_det.sh   models/rfdetr-base.onnx  data/dog.jpg   models/coco.names
./examples/run_video_det.sh   models/rfdetr-base.onnx  data/video.mp4 models/coco.names
./examples/run_camera_det.sh  models/rfdetr-base.onnx  0              models/coco.names
```

### Direct execution

```bash
# image
./build/examples/example_image_det models/rfdetr-base.onnx data/dog.jpg models/coco.names

# video (output saved next to input as <stem>_rfdetr.mp4)
./build/examples/example_video_det models/rfdetr-base.onnx data/video.mp4 models/coco.names

# camera (device index or /dev/videoN path)
./build/examples/example_camera_det models/rfdetr-base.onnx 0 models/coco.names
```

## Example code structure

Each example follows the same pattern:

```cpp
#include "rfdetrs/rfdetrs.hpp"
using namespace rfdetrs::det;

RFDETRDetector det(modelPath, labelsPath, BackendType::CPU);
cv::Mat img = cv::imread(imagePath);
auto results = det.detect(img, 0.5f, 300);
det.drawDetections(img, results);
cv::imwrite(outputPath, img);
```

## Output

Results are saved alongside the input file:
- Image: `<stem>_rfdetr.jpg` + `<stem>_rfdetr.json`
- Video: `<stem>_rfdetr.mp4`
- Camera snapshot: `snapshot_<timestamp>.jpg`
