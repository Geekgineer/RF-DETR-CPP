#!/usr/bin/env bash
# Run the video detection example
set -euo pipefail
cd "$(dirname "$0")/.."
exec ./build/examples/example_video_det "$@"
