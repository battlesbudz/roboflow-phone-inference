"""One-off export: RF-DETR Nano (Roboflow) -> ONNX fp16, with verification.

Run in CI via .github/workflows/export-rfdetr.yml (workflow_dispatch).
It must NOT be run to produce production artifacts from anywhere else:
the committed model file's provenance is exactly this script + the
export workflow + the provenance JSON it writes.

Pipeline:
  1. Instantiate Roboflow's RFDETRNano() (downloads the official COCO
     pretrained checkpoint from HuggingFace, roboflow org) and export
     to ONNX fp32 with Roboflow's own rfdetr export code.
  2. Convert the fp32 ONNX to fp16 with onnxconverter-common's float16
     converter (weights only; inputs/outputs stay fp32). RF-DETR Nano is ~30.5M
     params, so fp32 is ~122 MB -- over GitHub's 100 MB file limit.
     fp16 lands at ~61 MB and is a mechanical precision conversion,
     not a retrain.
  3. Verify fp32 vs fp16 on test-bus.jpg with verify_rfdetr.py's decode
     (Roboflow's contract): same top detections, scores within 0.05,
     and report latency for both.
  4. Write RFDETR_PROVENANCE.json and copy the fp16 ONNX to
     rf-detr-nano.onnx at the repo root. The workflow commits both.
"""

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    import onnx
    from onnxconverter_common import float16

    import verify_rfdetr

    root = Path(__file__).resolve().parent
    export_dir = root / "rfdetr_export"
    export_dir.mkdir(exist_ok=True)

    # 1. Export fp32 with Roboflow's own code.
    import importlib.metadata

    try:
        rfdetr_version = importlib.metadata.version("rfdetr")
    except Exception:
        rfdetr_version = "unknown"
    print(f"rfdetr version: {rfdetr_version}")
    from rfdetr import RFDETRNano

    model = RFDETRNano()
    model.export(str(export_dir))
    onnx_files = sorted(export_dir.glob("*.onnx"))
    if len(onnx_files) != 1:
        print(f"FAIL: expected exactly one exported ONNX, got {onnx_files}")
        sys.exit(1)
    fp32_path = onnx_files[0]
    print(f"exported fp32: {fp32_path} ({fp32_path.stat().st_size / 1e6:.1f} MB)")

    # 2. Convert to fp16 (weights only).
    fp16_path = root / "rf-detr-nano.onnx"
    m = onnx.load(str(fp32_path))
    m16 = float16.convert_float_to_float16(m, keep_io_types=True)
    onnx.save(m16, str(fp16_path))
    size_mb = fp16_path.stat().st_size / 1e6
    print(f"converted fp16: {fp16_path} ({size_mb:.1f} MB)")
    if size_mb > 95:
        print(f"FAIL: fp16 model {size_mb:.1f} MB is too close to GitHub's 100 MB file limit")
        sys.exit(1)

    # 3. Verify fp32 vs fp16 on the bundled test image.
    image = root / "test-bus.jpg"
    fp32_res = verify_rfdetr.run_model(str(fp32_path), str(image), threshold=0.3)
    fp16_res = verify_rfdetr.run_model(str(fp16_path), str(image), threshold=0.3)
    fp32_labels = [d["label"] for d in fp32_res["detections"]]
    fp16_labels = [d["label"] for d in fp16_res["detections"]]
    print(f"fp32 detections ({len(fp32_labels)}): {fp32_labels[:10]}")
    print(f"fp16 detections ({len(fp16_labels)}): {fp16_labels[:10]}")
    print(f"fp32 latency_ms mean: {fp32_res['latency_ms']['mean']}")
    print(f"fp16 latency_ms mean: {fp16_res['latency_ms']['mean']}")
    if "bus" not in fp32_labels or "bus" not in fp16_labels:
        print("FAIL: 'bus' not detected by fp32 and/or fp16 on test-bus.jpg")
        sys.exit(1)
    # Same top-5 labels in the same order, scores within 0.05.
    for a, b in zip(fp32_res["detections"][:5], fp16_res["detections"][:5]):
        if a["label"] != b["label"] or abs(a["confidence"] - b["confidence"]) > 0.05:
            print(f"FAIL: fp32/fp16 top-5 mismatch: {a} vs {b}")
            sys.exit(1)
    print("fp32/fp16 parity OK")

    # 4. Provenance.
    import onnxruntime as ort

    sess = ort.InferenceSession(str(fp16_path), providers=["CPUExecutionProvider"])
    provenance = {
        "model": "RF-DETR Nano (RFDETRNano), COCO pretrained",
        "source": "Roboflow rfdetr package (github.com/roboflow/rf-detr), "
                  "official COCO checkpoint downloaded by RFDETRNano() from HuggingFace (roboflow org)",
        "rfdetr_version": rfdetr_version,
        "export": "model.export(output_dir) with Roboflow's rfdetr ONNX exporter (opset 17, fp32), "
                  "then onnxconverter-common float16 converter (weights only, keep_io_types=True)",
        "license": "Apache 2.0 (RF-DETR-N is Apache 2.0 per roboflow/rf-detr README benchmarks)",
        "file": "rf-detr-nano.onnx",
        "sha256": sha256_of(str(fp16_path)),
        "size_bytes": fp16_path.stat().st_size,
        "input": {"name": sess.get_inputs()[0].name,
                  "shape": [int(d) for d in sess.get_inputs()[0].shape]},
        "outputs": [{"name": o.name, "shape": [int(d) for d in o.shape]}
                    for o in sess.get_outputs()],
        "class_layout": "logits (Q, 91); slot index = COCO category ID (sparse 1-90); "
                        "official COCO weights: final slot is the real class 90, not background; "
                        "decode keeps every slot, unmapped slots never emit",
        "decode": "per-class sigmoid, top-Q pairs, score > 0.3, no NMS; "
                  "preprocess: direct resize, /255, ImageNet mean/std, NCHW",
        "verification": {
            "image": "test-bus.jpg",
            "fp32": {"detections": fp32_labels[:10],
                     "latency_ms_mean": fp32_res["latency_ms"]["mean"]},
            "fp16": {"detections": fp16_labels[:10],
                     "latency_ms_mean": fp16_res["latency_ms"]["mean"]},
            "parity": "top-5 labels identical, scores within 0.05",
        },
        "exported_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    prov_path = root / "RFDETR_PROVENANCE.json"
    prov_path.write_text(json.dumps(provenance, indent=2))
    print(f"wrote {prov_path}")
    print("EXPORT OK")


if __name__ == "__main__":
    main()
