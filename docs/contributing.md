# Contributing

Contributions are welcome. Please open an issue or pull request on
[GitHub](https://github.com/Geekgineer/RF-DETR-CPP).

## Development Setup

```bash
git clone https://github.com/Geekgineer/RF-DETR-CPP.git
cd RF-DETR-CPP
cmake -B build -DONNXRUNTIME_DIR=... -DBUILD_TESTS=ON
cmake --build build --parallel $(nproc)
```

## Running Tests

```bash
# Detection parity
bash tests/test_detection.sh

# Segmentation structural
bash tests/test_segmentation.sh

# Keypoint structural
bash tests/test_keypoint.sh
```

## License

All contributions must be compatible with AGPL-3.0-only.
