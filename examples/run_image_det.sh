#!/usr/bin/env bash
# Run the image detection example
set -euo pipefail
cd "$(dirname "$0")/.."
exec ./build/examples/example_image_det "$@"
