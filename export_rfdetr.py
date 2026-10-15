"""One-off export: RF-DETR Nano (Roboflow) -> ONNX INT8 (QDQ), with verification.

Run in CI via .github/workflows/export-rfdetr.yml (workflow_dispatch).
It must NOT be run to produce production artifacts from anywhere else:
the committed model file's provenance is exactly this script + the
export workflow + the provenance JSON it writes.

Pipeline:
  1. Instantiate Roboflow's RFDETRNano() (downloads the official COCO
     pretrained checkpoint from HuggingFace, roboflow org) and export
     to ONNX with quantization="int8" using Roboflow's own rfdetr
     export code. INT8 uses static QDQ (QuantizeLinear /
     DequantizeLinear): fp32 in/out, int8 interior. The fp32 graph
     stays on disk as the accuracy baseline.
  2. Verify fp32 vs INT8 on test-bus.jpg with verify_rfdetr.py's decode
     (Roboflow's contract): same SET of top-5 labels, scores within
     0.1. INT8 is a quantised model, so the tolerance is looser than
     fp16 parity would be -- but a garbage quantisation still fails.
  3. Write RFDETR_PROVENANCE.json and copy the INT8 ONNX to
     rf-detr-nano.onnx at the repo root. The workflow commits both.

Why INT8 and not fp16: the fp16 converter (onnxconverter-common)
produces a graph ONNX Runtime rejects -- it inserts Cast nodes whose
output types disagree with the graph's value_info, and neither
boundary-Cast surgery nor shape-inference refresh fixes it. The
rfdetr exporter's own INT8 QDQ path is self-consistent by
construction (the exporter places the quantize ops itself), and at
~30 MB it is well under GitHub's 100 MB file limit (fp32 is ~122 MB).

Calibration: activation ranges come from variants of the bundled
test-bus.jpg (crops, flip, brightness, scales). It is one photo, so
the ranges are approximate -- the fp32-vs-INT8 parity check in step 2
is the real gate.
"""

import hashlib
import json
import sys
import time
from pathlib import Path


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def make_calibration_images(source_path, out_dir):
    """Variants of one photo for INT8 activation-range calibration."""
    from PIL import Image, ImageEnhance

    out_dir.mkdir(exist_ok=True)
    img = Image.open(source_path).convert("RGB")
    w, h = img.size
    variants = {"orig": img}
    variants["flip"] = img.transpose(Image.FLIP_LEFT_RIGHT)
    variants["bright_low"] = ImageEnhance.Brightness(img).enhance(0.7)
    variants["bright_high"] = ImageEnhance.Brightness(img).enhance(1.3)
    # Quadrant + center crops, resized back to full size.
    boxes = {
        "crop_tl": (0, 0, w // 2, h // 2),
        "crop_tr": (w // 2, 0, w, h // 2),
        "crop_bl": (0, h // 2, w // 2, h),
        "crop_br": (w // 2, h // 2, w, h),
        "crop_center": (w // 4, h // 4, 3 * w // 4, 3 * h // 4),
    }
    for name, box in boxes.items():
        variants[name] = img.crop(box).resize((w, h), Image.BILINEAR)
    variants["small"] = img.resize((w // 2, h // 2), Image.BILINEAR).resize((w, h), Image.BILINEAR)
    variants["gray"] = img.convert("L").convert("RGB")
    for name, v in variants.items():
        v.save(out_dir / f"{name}.jpg", "JPEG", quality=90)
    return sorted(out_dir.glob("*.jpg"))


def main():
    import verify_rfdetr

    root = Path(__file__).resolve().parent
    export_dir = root / "rfdetr_export"
    export_dir.mkdir(exist_ok=True)

    # 1. Export fp32 + INT8 with Roboflow's own code.
    import importlib.metadata

    try:
        rfdetr_version = importlib.metadata.version("rfdetr")
    except Exception:
        rfdetr_version = "unknown"
    print(f"rfdetr version: {rfdetr_version}")
    calib_files = make_calibration_images(root / "test-bus.jpg", root / "calib")
    print(f"calibration images: {len(calib_files)}")

    from rfdetr import RFDETRNano

    model = RFDETRNano()
    model.export(
        str(export_dir),
        format="onnx",
        quantization="int8",
        calibration_data=str(root / "calib"),
    )
    onnx_files = sorted(export_dir.glob("*.onnx"))
    print(f"exported files: {[(p.name, round(p.stat().st_size / 1e6, 1)) for p in onnx_files]}")
    if len(onnx_files) < 2:
        print(f"FAIL: expected fp32 + int8 ONNX files, got {[p.name for p in onnx_files]}")
        sys.exit(1)
    int8_candidates = [p for p in onnx_files if "int8" in p.name.lower()]
    if int8_candidates:
        int8_path = int8_candidates[0]
    else:
        # Fall back to the smaller file (INT8 is ~1/4 the size of fp32).
        int8_path = min(onnx_files, key=lambda p: p.stat().st_size)
    fp32_candidates = [p for p in onnx_files if p != int8_path]
    fp32_path = fp32_candidates[0]
    print(f"fp32: {fp32_path.name} ({fp32_path.stat().st_size / 1e6:.1f} MB)")
    print(f"int8: {int8_path.name} ({int8_path.stat().st_size / 1e6:.1f} MB)")

    out_path = root / "rf-detr-nano.onnx"
    out_path.write_bytes(int8_path.read_bytes())
    size_mb = out_path.stat().st_size / 1e6
    if size_mb > 95:
        print(f"FAIL: int8 model {size_mb:.1f} MB is too close to GitHub's 100 MB file limit")
        sys.exit(1)

    # 2. Verify fp32 vs INT8 on the bundled test image.
    image = root / "test-bus.jpg"
    fp32_res = verify_rfdetr.run_model(str(fp32_path), str(image), threshold=0.3)
    int8_res = verify_rfdetr.run_model(str(out_path), str(image), threshold=0.3)
    fp32_labels = [d["label"] for d in fp32_res["detections"]]
    int8_labels = [d["label"] for d in int8_res["detections"]]
    print(f"fp32 detections ({len(fp32_labels)}): {fp32_labels[:10]}")
    print(f"int8 detections ({len(int8_labels)}): {int8_labels[:10]}")
    print(f"fp32 latency_ms mean: {fp32_res['latency_ms']['mean']}")
    print(f"int8 latency_ms mean: {int8_res['latency_ms']['mean']}")
    if "bus" not in fp32_labels or "bus" not in int8_labels:
        print("FAIL: 'bus' not detected by fp32 and/or int8 on test-bus.jpg")
        sys.exit(1)
    # INT8 parity: same SET of top-5 labels, scores within 0.1.
    # (Order may shuffle when two detections score close together.)
    fp32_top = {d["label"]: d["confidence"] for d in fp32_res["detections"][:5]}
    int8_top = {d["label"]: d["confidence"] for d in int8_res["detections"][:5]}
    if set(fp32_top) != set(int8_top):
        print(f"FAIL: fp32/int8 top-5 label sets differ: {sorted(fp32_top)} vs {sorted(int8_top)}")
        sys.exit(1)
    for label in fp32_top:
        if abs(fp32_top[label] - int8_top[label]) > 0.1:
            print(f"FAIL: fp32/int8 score drift on '{label}': "
                  f"{fp32_top[label]} vs {int8_top[label]}")
            sys.exit(1)
    print("fp32/int8 parity OK")

    # 3. Provenance.
    import onnxruntime as ort

    sess = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
    provenance = {
        "model": "RF-DETR Nano (RFDETRNano), COCO pretrained",
        "source": "Roboflow rfdetr package (github.com/roboflow/rf-detr), "
                  "official COCO checkpoint downloaded by RFDETRNano() from HuggingFace (roboflow org)",
        "rfdetr_version": rfdetr_version,
        "export": "model.export(format='onnx', quantization='int8', calibration_data=<test-bus.jpg variants>) "
                  "with Roboflow's rfdetr ONNX exporter (opset 17). Static QDQ: fp32 in/out, int8 interior.",
        "license": "Apache 2.0 (RF-DETR-N is Apache 2.0 per roboflow/rf-detr README benchmarks)",
        "file": "rf-detr-nano.onnx",
        "sha256": sha256_of(str(out_path)),
        "size_bytes": out_path.stat().st_size,
        "input": {"name": sess.get_inputs()[0].name,
                  "shape": [int(d) for d in sess.get_inputs()[0].shape]},
        "outputs": [{"name": o.name, "shape": [int(d) for d in o.shape]}
                    for o in sess.get_outputs()],
        "class_layout": "logits (Q, 91); slot index = COCO category ID (sparse 1-90); "
                        "official COCO weights: final slot is the real class 90, not background; "
                        "decode keeps every slot, unmapped slots never emit",
        "decode": "per-class sigmoid, top-Q pairs, score > 0.3, no NMS; "
                  "preprocess: direct resize, /255, ImageNet mean/std, NCHW",
        "calibration": f"{len(calib_files)} variants of test-bus.jpg (crops, flip, brightness, scales, gray)",
        "verification": {
            "image": "test-bus.jpg",
            "fp32": {"detections": fp32_labels[:10],
                     "latency_ms_mean": fp32_res["latency_ms"]["mean"]},
            "int8": {"detections": int8_labels[:10],
                     "latency_ms_mean": int8_res["latency_ms"]["mean"]},
            "parity": "same top-5 label set, scores within 0.1",
        },
        "exported_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    prov_path = root / "RFDETR_PROVENANCE.json"
    prov_path.write_text(json.dumps(provenance, indent=2))
    print(f"wrote {prov_path}")
    print("EXPORT OK")


if __name__ == "__main__":
    main()
