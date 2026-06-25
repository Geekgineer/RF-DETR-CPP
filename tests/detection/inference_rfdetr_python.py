#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2024 Geekgineer
# SPDX-License-Identifier: AGPL-3.0-only
"""RF-DETR ONNX reference inference (Python / onnxruntime).

This script produces results/results_python.json, which is compared against
the C++ output in the parity GTest.

IMPORTANT: preprocessing must be byte-for-byte identical to the C++
implementation.  We use cv2.resize (INTER_LINEAR) — NOT PIL.Image.resize —
so the bilinear kernel matches OpenCV on both sides.

Postprocessing mirrors rfdetr/src/rfdetr/models/postprocess.py exactly:
  prob = sigmoid(labels)           # [1, N, C]
  flat = prob.reshape(-1)          # [N*C]
  top_k indices                    # (query_idx, class_idx) pairs
  boxes = cxcywh_to_xyxy(dets)    # xyxy normalized
  boxes *= [W, H, W, H]           # scale to original pixel size
"""

import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).parent
DATA_DIR   = SCRIPT_DIR / "data" / "images"
MODEL_DIR  = SCRIPT_DIR / "models"
OUT_DIR    = SCRIPT_DIR / "results"
CFG_PATH   = SCRIPT_DIR / "inference_config.json"

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ---------------------------------------------------------------------------
# Preprocessing — must match preprocessing.hpp resizeAndNormalize()
# ---------------------------------------------------------------------------
def preprocess(bgr_image: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    """Return float32 CHW blob normalised with ImageNet mean/std."""
    H, W = target_hw
    # Stretch resize — cv2.INTER_LINEAR matches torch F.resize default
    resized = cv2.resize(bgr_image, (W, H), interpolation=cv2.INTER_LINEAR)
    # BGR → RGB, /255, normalize
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb = (rgb - IMAGENET_MEAN) / IMAGENET_STD
    # HWC → CHW, add batch dim
    return rgb.transpose(2, 0, 1)[np.newaxis]  # [1, 3, H, W]


# ---------------------------------------------------------------------------
# Postprocessing — mirrors postprocess.py _select_topk + _gather_and_scale_boxes
# ---------------------------------------------------------------------------
def postprocess(dets: np.ndarray, labels: np.ndarray,
                orig_hw: tuple[int, int], conf: float, top_k: int) -> list[dict]:
    """
    dets:   [1, N, 4]  cxcywh normalized
    labels: [1, N, C]  class logits
    """
    orig_H, orig_W = orig_hw
    N = labels.shape[1]
    C = labels.shape[2]

    # sigmoid
    prob = 1.0 / (1.0 + np.exp(-labels[0]))  # [N, C]

    # flatten and topk
    flat = prob.reshape(-1)  # [N*C]
    k = min(top_k, len(flat))
    topk_flat_idx = np.argpartition(flat, -k)[-k:]
    topk_flat_idx = topk_flat_idx[np.argsort(-flat[topk_flat_idx])]

    results = []
    for flat_idx in topk_flat_idx:
        score = float(flat[flat_idx])
        if score < conf:
            break  # sorted descending
        qi = int(flat_idx) // C
        ci = int(flat_idx) % C

        cx, cy, bw, bh = dets[0, qi]
        x1 = (cx - bw / 2) * orig_W
        y1 = (cy - bh / 2) * orig_H
        x2 = (cx + bw / 2) * orig_W
        y2 = (cy + bh / 2) * orig_H

        x1 = max(0, min(int(x1), orig_W - 1))
        y1 = max(0, min(int(y1), orig_H - 1))
        x2 = max(0, min(int(x2), orig_W))
        y2 = max(0, min(int(y2), orig_H))

        results.append({
            "class_id":   ci,
            "confidence": score,
            "bbox": {
                "left":   x1,
                "top":    y1,
                "width":  x2 - x1,
                "height": y2 - y1,
            },
        })
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def load_config() -> dict:
    if CFG_PATH.exists():
        with open(CFG_PATH) as f:
            return json.load(f)
    return {"conf": 0.5, "top_k": 300}


def identify_outputs(session: ort.InferenceSession) -> tuple[str, str]:
    """Return (box_name, logit_name) by inspecting output shapes."""
    box_name = logit_name = None
    for meta in session.get_outputs():
        shape = meta.shape
        if len(shape) >= 3:
            last_dim = shape[-1]
            if last_dim == 4 and box_name is None:
                box_name = meta.name
            elif isinstance(last_dim, int) and last_dim > 4 and logit_name is None:
                logit_name = meta.name
    if box_name is None or logit_name is None:
        raise RuntimeError(
            "Cannot identify box/logit outputs from session output shapes. "
            f"Outputs: {[m.name for m in session.get_outputs()]}"
        )
    return box_name, logit_name


def run_model(model_path: Path, cfg: dict) -> dict:
    print(f"\n{'='*60}")
    print(f"Model: {model_path}")

    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(str(model_path), sess_opts,
                                   providers=["CPUExecutionProvider"])

    # Input shape
    inp_meta = session.get_inputs()[0]
    inp_shape = inp_meta.shape  # [1, 3, H, W]
    target_H, target_W = int(inp_shape[2]), int(inp_shape[3])
    print(f"Input: {inp_shape}")

    box_name, logit_name = identify_outputs(session)
    print(f"Box output: '{box_name}'  Logit output: '{logit_name}'")

    conf  = float(cfg.get("conf",  0.5))
    top_k = int(cfg.get("top_k", 300))

    image_files = sorted(
        p for p in DATA_DIR.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    )
    if not image_files:
        print(f"[WARNING] No images found in {DATA_DIR}")
        return {"results": []}

    model_name = model_path.stem
    model_results = {
        "weights_path": str(model_path),
        "task": "detect",
        "results": [],
    }

    for img_path in image_files:
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            print(f"[SKIP] Cannot load {img_path}")
            continue

        orig_H, orig_W = bgr.shape[:2]
        blob = preprocess(bgr, (target_H, target_W))

        outputs = session.run([box_name, logit_name], {inp_meta.name: blob})
        # outputs[0] is box, outputs[1] is logits (matched by name)
        dets_arr   = outputs[0]
        labels_arr = outputs[1]

        dets = postprocess(dets_arr, labels_arr, (orig_H, orig_W), conf, top_k)

        model_results["results"].append({
            "image_path": str(img_path),
            "inference_results": dets,
        })
        print(f"  {img_path.name}: {len(dets)} detections")

    return model_results


def main():
    cfg = load_config()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    model_files = sorted(MODEL_DIR.glob("*.onnx"))
    if not model_files:
        print(f"[ERROR] No .onnx models found in {MODEL_DIR}")
        print("Export a model first: python scripts/export_rfdetr_onnx.py")
        sys.exit(1)

    all_results = {}
    for model_path in model_files:
        model_name = model_path.stem
        all_results[model_name] = run_model(model_path, cfg)

    out_path = OUT_DIR / "results_python.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
