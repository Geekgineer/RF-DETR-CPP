#!/usr/bin/env python3
"""Export RF-DETR model to ONNX for C++ inference.

Usage:
    python scripts/export_rfdetr_onnx.py --variant nano [--weights rfdetr-nano.pt] [--out-dir models/]
    python scripts/export_rfdetr_onnx.py --variant large --weights rfdetr-large.pt --out-dir models/

The exported file is named <out-dir>/inference_model.onnx.
Rename it (e.g. rfdetr-large.onnx) if you want to keep multiple variants.

Supported variants (Apache-2.0):
    nano, small, base, medium, large
"""

import argparse
import shutil
from pathlib import Path


VARIANT_MAP = {
    "nano":   "RFDETRNano",
    "small":  "RFDETRSmall",
    "base":   "RFDETRBase",
    "medium": "RFDETRMedium",
    "large":  "RFDETRLarge",
}


def main():
    parser = argparse.ArgumentParser(description="Export RF-DETR to ONNX")
    parser.add_argument("--variant", default="large",
                        choices=list(VARIANT_MAP.keys()),
                        help="Model variant (default: large)")
    parser.add_argument("--weights", default=None,
                        help="Path to .pt checkpoint (downloads default if omitted)")
    parser.add_argument("--out-dir", default="models",
                        help="Output directory (default: models/)")
    args = parser.parse_args()

    try:
        import rfdetr as _rfdetr_pkg
    except ImportError:
        raise SystemExit(
            "rfdetr not installed. Install with:\n"
            "  pip install rfdetr[onnx]\n"
            "or from the rf-detr/ source:\n"
            "  pip install -e 'rf-detr/[onnx]'"
        )

    cls_name = VARIANT_MAP[args.variant]
    # Import the variant class
    cls = getattr(_rfdetr_pkg, cls_name)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Construct model (downloads pretrained weights if not provided)
    kwargs = {}
    if args.weights:
        kwargs["pretrain_weights"] = args.weights

    print(f"Loading {cls_name} ...")
    model = cls(**kwargs)

    # RF-DETR exports to a subdirectory called "output/" by default.
    # We redirect that to a temp location and then move the file.
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        print(f"Exporting to {tmp}/inference_model.onnx ...")
        model.export(output_dir=tmp)

        # rfdetr v1.8 exports as "rfdetr-<variant>.onnx"; older as "inference_model.onnx"
        src = Path(tmp) / "inference_model.onnx"
        if not src.exists():
            candidates = list(Path(tmp).rglob("*.onnx"))
            if not candidates:
                raise FileNotFoundError(
                    f"No .onnx file found under {tmp}. Check rfdetr version."
                )
            src = candidates[0]

        dst = out_dir / f"rfdetr-{args.variant}.onnx"
        shutil.copy2(src, dst)

    print(f"\nExported: {dst}")
    print(f"Input shape : 1 x 3 x {model.model.resolution} x {model.model.resolution}")
    print("Outputs     : dets [1,N,4]  labels [1,N,C]")


if __name__ == "__main__":
    main()
