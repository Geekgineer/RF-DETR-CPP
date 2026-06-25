# Release Guide for RFDETRs-CPP

This guide explains how to prepare and publish a release for RFDETRs-CPP.

## Pre-Release Checklist

### 1. Verify All Tests Pass

```bash
# build with tests enabled
cmake -B build -DCMAKE_BUILD_TYPE=Release \
      -DONNXRUNTIME_DIR=/path/to/onnxruntime \
      -DBUILD_TESTS=ON
cmake --build build -j$(nproc)

# place test assets then run
cp models/rfdetr-base.onnx tests/detection/models/
cp data/dog.jpg            tests/detection/data/images/
./run_test.sh
```

Expected: 4/4 GTest assertions pass.

### 2. Run Benchmarks

```bash
cmake -B benchmarks/build -S benchmarks \
      -DONNXRUNTIME_DIR=/path/to/onnxruntime -DCMAKE_BUILD_TYPE=Release
cmake --build benchmarks/build -j$(nproc)
./benchmarks/auto_bench.sh --backend cpu --mode image
```

### 3. Update Version Numbers

Update version in:
- `CMakeLists.txt` — `project(RFDETRs-CPP VERSION x.y.z)`
- `include/rfdetrs/core/version.hpp` — `RFDETRS_VERSION_*` constants
- `README.md` — badges and documentation

### 4. Prepare Model Assets

```bash
# Export all supported variants for release testing
.venv/bin/python scripts/export_rfdetr_onnx.py --variant nano   --out-dir models/
.venv/bin/python scripts/export_rfdetr_onnx.py --variant small  --out-dir models/
.venv/bin/python scripts/export_rfdetr_onnx.py --variant base   --out-dir models/
.venv/bin/python scripts/export_rfdetr_onnx.py --variant medium --out-dir models/
.venv/bin/python scripts/export_rfdetr_onnx.py --variant large  --out-dir models/
```

## Creating the Release

### Step 1: Tag and push

```bash
git tag -a v1.0.0 -m "RFDETRs-CPP v1.0.0"
git push origin v1.0.0
```

### Step 2: Create GitHub Release

1. Go to GitHub → Releases → "Create new release"
2. **Tag:** `v1.0.0`
3. **Title:** "RFDETRs-CPP v1.0.0"
4. Use the template below for the description

## Release Notes Template

```markdown
## What's New in v1.0.0

### Features
- RF-DETR detection: Nano · Small · Base · Medium · Large (all Apache-2.0)
- CPU, CUDA, TensorRT execution providers via `BackendType` enum
- Image / Video / Camera / Batch inference modes
- Parity-tested against Python ORT reference (±2 px bbox, ±0.002 score)
- Docker images for CPU and GPU

### Supported Models

| Variant | Resolution | License |
|---------|-----------|---------|
| Nano    | 384×384   | Apache-2.0 |
| Small   | 512×512   | Apache-2.0 |
| Base    | 560×560   | Apache-2.0 |
| Medium  | 576×576   | Apache-2.0 |
| Large   | 704×704   | Apache-2.0 |

### Performance (CPU — Intel Core i7)

| Model | FPS | Avg latency |
|-------|-----|-------------|
| rfdetr-base | 2.51 | 399 ms |

### Requirements
- CMake 3.16+
- C++17 compiler
- OpenCV 4.5+
- ONNX Runtime 1.20.1+

### Contributors
- @Geekgineer
```

## Post-Release

### Verify CI/CD

After release:
1. GitHub Actions workflow passes (`build-cpu` + `test-suite`)
2. Tests can run from a clean checkout
3. README links are correct

## Troubleshooting

### Tests fail
1. Check that model files exist in `tests/detection/models/`
2. Check that test images exist in `tests/detection/data/images/`
3. Verify ORT version matches (1.20.1)

### CI build fails
1. Check ORT download URL in workflow (`ONNXRUNTIME_VERSION` env var)
2. Check that OpenCV is available on the runner (`libopencv-dev`)
3. Check CMake version (`cmake --version`)

## File Locations

| Purpose | Location |
|---------|----------|
| Export script | `scripts/export_rfdetr_onnx.py` |
| Test runner | `run_test.sh` |
| CI workflow | `.github/workflows/main.yml` |
| Benchmark tool | `benchmarks/rfdetr_benchmark.cpp` |
