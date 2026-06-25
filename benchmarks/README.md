# RFDETRs-CPP Benchmark Tool

Standalone benchmark for RF-DETR ONNX inference — measures latency statistics
(min / max / avg / median) and FPS across image, video, and batch modes.

## Building

```bash
cmake -B benchmarks/build -S benchmarks \
      -DONNXRUNTIME_DIR=/path/to/onnxruntime \
      -DCMAKE_BUILD_TYPE=Release
cmake --build benchmarks/build -j$(nproc)
```

Or use `auto_bench.sh` which auto-discovers ORT and builds:

```bash
./benchmarks/auto_bench.sh --backend cpu --mode image
```

## Usage

```
rfdetr_benchmark <model.onnx> <image_or_video> [labels.names]
                 [--backend cpu|cuda|trt]
                 [--mode image|video|batch]
                 [--warmup N]     (default 3)
                 [--iters N]      (default 10)
                 [--conf F]       (default 0.5)
                 [--topk N]       (default 300)
```

### Image benchmark

```bash
./benchmarks/build/rfdetr_benchmark models/rfdetr-base.onnx data/dog.jpg \
    models/coco.names --backend cpu --mode image --warmup 3 --iters 100
```

### Video benchmark

```bash
./benchmarks/build/rfdetr_benchmark models/rfdetr-base.onnx data/video.mp4 \
    models/coco.names --backend cpu --mode video --iters 200
```

### Automated sweep

`auto_bench.sh` discovers all `.onnx` files under `models/`, runs each, and
prints a summary table.

```bash
./benchmarks/auto_bench.sh --backend cpu --mode image --warmup 5 --iters 100
```

## Output format

```
=== Benchmark: rfdetr-base ===
Backend  : cpu
Mode     : image
Warmup   : 3 runs
Measured : 10 runs

  Min    : 388.2 ms
  Max    : 441.8 ms
  Avg    : 399.1 ms
  Median : 395.3 ms
  FPS    : 2.51
```

Results are also appended to `benchmarks/results/benchmark_results.csv`.

## Prerequisites

- CMake 3.16+
- C++17 compiler
- OpenCV 4.5+
- ONNX Runtime 1.20.1+

## Troubleshooting

**"No .onnx models found"** — check that `models/rfdetr-*.onnx` exists.
Export with `python scripts/export_rfdetr_onnx.py --variant base`.

**GPU not detected** — use the GPU-flavoured ORT package
(`onnxruntime-linux-x64-gpu-1.20.1.tgz`) and pass `--backend cuda`.
