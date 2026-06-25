#!/usr/bin/env python3
"""Convenience wrapper — export RF-DETR to ONNX into this directory.

Usage:
    cd models/
    python export_onnx.py                     # exports base → ./rfdetr-base.onnx
    python export_onnx.py --variant large     # exports large
    python export_onnx.py --variant nano --weights nano.pt
"""
import sys
from pathlib import Path

# Delegate to the canonical script so logic lives in one place
scripts_dir = Path(__file__).parent.parent / "scripts"
sys.path.insert(0, str(scripts_dir))

# Patch sys.argv so the canonical main() sees --out-dir models/
import sys as _sys
if "--out-dir" not in _sys.argv:
    _sys.argv += ["--out-dir", str(Path(__file__).parent)]

from export_rfdetr_onnx import main
main()
