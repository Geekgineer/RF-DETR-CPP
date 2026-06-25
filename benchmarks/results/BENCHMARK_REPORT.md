# RFDETRs-CPP Benchmark Report

Platform: Linux x86-64 · Intel Core i7 · single process  
ONNX Runtime: 1.20.1 · CPU EP

---

## Image inference (dog.jpg, 302×329)

| Date | Model | Backend | Warmup | Iters | Min (ms) | Avg (ms) | Max (ms) | Median (ms) | FPS |
|------|-------|---------|--------|-------|---------|---------|---------|------------|-----|
| 2026-06-24 | rfdetr-base | CPU | 3 | 10 | 388.2 | 399.1 | 441.8 | 395.3 | 2.51 |
| 2026-06-25 | rfdetr-base | CPU | 5 | 20 | 382.5 | 393.3 | 404.3 | 393.7 | 2.54 |

---

## Phase breakdown (2026-06-25, last iteration)

| Phase | Time (ms) | Share |
|-------|-----------|-------|
| Preprocess (resize + CHW normalize) | 1.02 | 0.3% |
| Inference (ORT session.Run) | 388.95 | 99.7% |
| Postprocess (logit filter + top-K decode) | 0.02 | <0.1% |

---

## Notes

- RF-DETR Base uses 560×560 static input shape; all images are stretch-resized.
- No NMS is applied — postprocess is sigmoid → flat top-K decode only.
- First run (excluded by warmup) includes JIT compilation overhead.
- GPU numbers will vary by hardware; use `--backend cuda` with a GPU ORT build.
- Postprocess was optimized from ~0.8ms to 0.02ms (40×) by pre-filtering in logit
  space: `sigmoid(x) >= t ⟺ x >= logit(t)`, eliminating exp() on all but a handful
  of the 300×91=27,300 candidate (query, class) pairs.
- The dominant cost is the ONNX model execution (transformer self-attention).
  For further speedup, enable CUDA EP (`--backend cuda`) or TensorRT FP16.
