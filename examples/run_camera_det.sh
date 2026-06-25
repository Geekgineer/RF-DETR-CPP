#!/usr/bin/env bash
# Run the camera detection example
set -euo pipefail
cd "$(dirname "$0")/.."
exec ./build/examples/example_camera_det "$@"
