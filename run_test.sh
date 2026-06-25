#!/usr/bin/env bash
# Quick test runner — delegates to tests/test_all.sh
set -euo pipefail
cd "$(dirname "$0")"
exec bash tests/test_all.sh "$@"
